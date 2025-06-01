import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Enable cuDNN auto-tuning for potentially faster GPU performance
torch.backends.cudnn.benchmark = True


#
# 1. LOAD DAILY CSV AND TRANSFORM INTO A TORCH TENSOR OF SHAPE (N, T_days, Fcnt)
#    — ANY NaN VALUES ARE FORWARD‐FILLED (AND BACKWARD‐FILLED IF AT THE VERY START)
#
def load_daily_tensor(csv_path):
    """
    1) Read raw CSV file containing multiple instruments (stocks) over time.
       We no longer drop rows with NaNs here; instead, we will forward‐fill missing values.
    2) Detect the date column (Date / date / datetime), parse it to datetime.
    3) Pivot to a wide DataFrame with index = datetime, columns = (instrument, feature).
    4) Forward‐fill any NaNs along the time‐axis; then backward‐fill the first row if needed.
    5) Quantile‐transform all values into a normal distribution.
    6) Return a tensor `data` of shape (N, T_days, Fcnt), plus:
         - instruments: list of N ticker names
         - features: list of Fcnt feature names
         - df_pivot: the wide DataFrame after pivoting & filling NaNs.
    """
    # 1.1. Read CSV without dropping NA immediately
    df = pd.read_csv(csv_path)

    # 1.2. Detect which column holds the date info and convert it
    if "Date" in df.columns:
        df["datetime"] = pd.to_datetime(df["Date"])
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"])
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        raise ValueError("No date column found in CSV (expected 'Date' or 'date' or 'datetime').")

    # 1.3. Pivot to a wide format: index = datetime, columns = (instrument, feature)
    df = df.rename(columns={"datetime": "datetime"})  # just to be explicit
    # We expect the raw CSV to have columns: [ datetime, instrument, <features like Open/High/Close/...> ]
    df_wide = df.set_index(["datetime", "instrument"])
    df_pivot = df_wide.unstack("instrument")  # now columns become MultiIndex: (feature, instrument)
    # Swap levels so that columns = (instrument, feature)
    df_pivot.columns = df_pivot.columns.swaplevel(0, 1)
    df_pivot.columns.names = ["instrument", "feature"]
    df_pivot = df_pivot.sort_index(axis=1)

    # 1.4. FORWARD‐FILL along the time axis (so any NaN at day t is replaced by day t-1).
    #      Then backward‐fill the very first rows if they still contain NaNs.
    df_pivot = df_pivot.ffill().bfill()

    # 1.5. Extract lists of instruments (N) and features (Fcnt)
    instruments = list(df_pivot.columns.levels[0])
    features = list(df_pivot.columns.levels[1])

    # 1.6. Convert wide DataFrame into a NumPy array of shape (T_days, N * Fcnt)
    T_days = len(df_pivot)
    N = len(instruments)
    Fcnt = len(features)
    arr = df_pivot.values.reshape(T_days, N * Fcnt)

    # 1.7. Quantile‐transform each “row” (all N*Fcnt values) to a normal distribution
    qt = QuantileTransformer(output_distribution="normal")
    arr_trans = qt.fit_transform(arr)

    # 1.8. Reshape into (T_days, N, Fcnt), then transpose to (N, T_days, Fcnt)
    data = torch.tensor(arr_trans.reshape(T_days, N, Fcnt).transpose(1, 0, 2), dtype=torch.float)

    return data, instruments, features, df_pivot


#
# 2. WEEKLY AGGREGATION (COARSE‐GRAINED) FROM DAILY DataFrame
#
def aggregate_weekly(df_pivot):
    """
    Given df_pivot (index=Datetime, columns=(instrument,feature)),
    resample to weekly frequency (Friday mean) and drop any resulting NaNs
    (there shouldn’t be any because we forward‐filled daily).
    Returns df_week with shape (T_weeks, N * Fcnt).
    """
    df_week = df_pivot.resample("W-FRI").mean().dropna()
    return df_week


#
# 3. SLIDING WINDOWS ON FINE‐GRAINED DATA
#
def sliding_windows(data, L, stride):
    """
    Build sliding windows of length L (timesteps) with step size = stride on the fine‐grained tensor.
    - data: shape (N, T, Fcnt)
    - We slide along the T axis in steps of `stride`. Each window is (N, L, Fcnt) then flattened to (N, L*Fcnt).
    Returns:
      - ws: Tensor of shape (N, num_windows, L*Fcnt)
      - starts: list of starting indices
    """
    N, T, F = data.shape
    ws = []
    starts = []
    for i in range(0, T - L + 1, stride):
        block = data[:, i : i + L, :].reshape(N, -1)  # (N, L*Fcnt)
        ws.append(block)
        starts.append(i)
    if len(ws) == 0:
        raise ValueError(f"No sliding windows formed (T={T}, L={L}, stride={stride})")
    return torch.stack(ws, dim=1), starts  # => (N, num_windows, L*Fcnt)


#
# 4. MODEL DEFINITIONS: FineEncoder, CoarseEncoder, Regressor
#
class FineEncoder(nn.Module):
    """
    Fine‐grained encoder: input dimension = L*Fcnt, output dimension = D (normalized).
    In the paper, this is h^t_f = Enc_f(x^t_f; ω_f).
    """

    def __init__(self, in_dim, D=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, D),
            nn.ReLU(inplace=True),
            nn.Linear(D, D),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (batch_size, in_dim). Return normalized embedding (batch_size, D).
        return F.normalize(self.net(x), dim=1)


class CoarseEncoder(nn.Module):
    """
    Coarse‐grained encoder: input dimension = Fcnt, output dimension = D (normalized).
    In the paper, this is h^t_c = Enc_c(x^t_c; ω_c).
    """

    def __init__(self, in_dim, D=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, D),
            nn.ReLU(inplace=True),
            nn.Linear(D, D),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class Regressor(nn.Module):
    """
    Regression head: takes a D‐dim embedding and outputs a scalar via tanh in [−1,1].
    In the paper, they use LeakyReLU(W h + b) for pred; here we choose a simpler tanh‐linear.
    """

    def __init__(self, D=128):
        super().__init__()
        self.linear = nn.Linear(D, 1)

    def forward(self, x):
        # x: (batch_size, D) → output (batch_size,)
        return torch.tanh(self.linear(x)).squeeze()


#
# 5. PRE‐TRAINING STAGE (CONTRASTIVE + SELF‐DISTILLATION → GDA + MSD)
#    (Simplified: we omit memory uploads/updates, just train the two encoders.)
#
def pretrain_stage(
    enc_f,
    enc_c,
    fm_tr,
    cm_tr,
    M_f,
    M_c,
    p=0.3,
    q=0.1,
    tau=0.5,
    lr=1e-4,
    epochs=500,
    batch_size=1024,
    early_stop=10,
    device="cpu",
):
    """
    Input:
      - enc_f: FineEncoder instance
      - enc_c: CoarseEncoder instance
      - fm_tr: fine‐grained windows tensor, shape (N, W_train, L*Fcnt)
      - cm_tr: coarse‐grained tensor, shape (N, W_train, Fcnt)
      - M_f, M_c: (unused dummy) memory matrices (M_size x D)
    Output:
      - (M_f, M_c) unchanged (we skip the actual memory updates here)
    """

    # Build DataLoader over (fine_window, coarse_vector) pairs
    ds = TensorDataset(fm_tr, cm_tr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    # Only optimize enc_f and enc_c parameters
    optimizer = torch.optim.AdamW(
        list(enc_c.parameters()) + list(enc_f.parameters()), lr=lr, weight_decay=1e-3
    )

    best_loss = float("inf")
    no_improve = 0

    for ep in range(epochs):
        total_loss = 0.0
        count = 0

        for x_f_batch, x_c_batch in loader:
            x_f = x_f_batch.to(device)  # (batch_size, L*Fcnt)
            x_c = x_c_batch.to(device)  # (batch_size, Fcnt)

            # 5.1. Forward pass: get embeddings from both encoders
            h_f = enc_f(x_f)  # (batch_size, D)
            h_c = enc_c(x_c)  # (batch_size, D)

            # 5.2. GDA: randomly replace some dims of h_c with corresponding dims of h_f
            mask = torch.rand_like(h_c) < p  # boolean mask with probability p
            h_tilde = torch.where(mask, h_f, h_c)

            # 5.3. Self‐Distillation (MSD) via KL divergences
            loss_f = F.kl_div(
                F.log_softmax(h_f / tau, dim=1),
                F.softmax(h_f.detach() / tau, dim=1),
                reduction="batchmean",
            )
            loss_c = F.kl_div(
                F.log_softmax(h_tilde / tau, dim=1),
                F.softmax(h_tilde.detach() / tau, dim=1),
                reduction="batchmean",
            )

            loss = loss_f + loss_c

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg = total_loss / count
        print(f"Pretrain Ep {ep+1}/{epochs}, Loss={avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop:
                print(f"→ Pretrain early stopping at epoch {ep+1}")
                break

    return M_f, M_c  # unchanged in this simplified version


#
# 6. FINE‐TUNING STAGE (REGRESSION)
#    (We only include the two predictive MSE terms; SI and CT losses are omitted.)
#
def finetune_stage(
    enc_f,
    enc_c,
    reg,
    fm_tr,
    cm_tr,
    labels_tr,
    lambdas=(1, 1, 3, 1, 3, 1, 3),
    lr=1e-3,
    epochs=500,
    batch_size=512,
    early_stop=30,
    device="cpu",
):
    """
    Input:
      - enc_f, enc_c: pretrained encoders from Stage 1
      - reg: regression head
      - fm_tr: fine‐grained windows (N, W_train, L*Fcnt)
      - cm_tr: coarse‐grained data (N, W_train, Fcnt)
      - labels_tr: true weekly returns (N, W_train)
    We optimize:
      L = λ3 * MSE(reg(enc_c(x_c)), y)  +  λ5 * MSE(reg(enc_f(x_f)), y)
    (We ignore triplet and InfoNCE terms here for brevity.)
    """
    λ1, λ2, λ3, λ4, λ5, λ6, λ7 = lambdas
    ds = TensorDataset(fm_tr, cm_tr, labels_tr)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        list(enc_c.parameters()) + list(enc_f.parameters()) + list(reg.parameters()),
        lr=lr,
        weight_decay=1e-3,
    )

    mse_loss = nn.MSELoss()
    best_loss = float("inf")
    no_improve = 0

    for ep in range(epochs):
        total_loss = 0.0
        count = 0

        for x_f_batch, x_c_batch, y_batch in loader:
            x_f = x_f_batch.to(device)  # (batch_size, L*Fcnt)
            x_c = x_c_batch.to(device)  # (batch_size, Fcnt)
            y = y_batch.to(device)      # (batch_size,)

            # 6.1. Compute embeddings
            h_f = enc_f(x_f)  # (batch_size, D)
            h_c = enc_c(x_c)  # (batch_size, D)

            # 6.2. Regression outputs
            y_from_c = reg(h_c)  # (batch_size,)
            y_from_f = reg(h_f)  # (batch_size,)

            # 6.3. MSE losses
            Lr1 = λ3 * mse_loss(y_from_c, y)
            Lr2 = λ5 * mse_loss(y_from_f, y)
            loss = Lr1 + Lr2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg = total_loss / count
        print(f"Finetune Ep {ep+1}/{epochs}, Loss={avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop:
                print(f"→ Fine-tune early stopping at epoch {ep+1}")
                break


#
# 7. EVALUATION METRICS ON TEST SET
#
def evaluate(reg, enc_c, fm, cm, labels, device="cpu"):
    """
    Compute:
     - MSE, MAE between predicted and true returns
     - Accuracy, F1, AUC for sign of direction.
    fm: fine‐grained windows (unused for evaluation here, but included to match signature)
    cm: coarse‐grained data (N, W_test, Fcnt)
    labels: true returns (N, W_test)
    """
    N, W, _ = fm.shape
    all_true = []
    all_pred = []
    all_clf = []

    for t in range(W):
        x_c = cm[:, t, :].to(device)           # (N, Fcnt)
        yhat = reg(enc_c(x_c)).detach().cpu().numpy()  # (N,)
        ytrue = labels[:, t].cpu().numpy()  # (N,)

        correct_dir = (ytrue * yhat) > 0  # boolean array, (N,)
        all_true.append(ytrue)
        all_pred.append(yhat)
        all_clf.append(correct_dir)

    y_t = np.concatenate(all_true)    # shape (N*W,)
    y_p = np.concatenate(all_pred)    # shape (N*W,)
    cfs = np.concatenate(all_clf)     # shape (N*W,)

    return {
        "MSE": mean_squared_error(y_t, y_p),
        "MAE": mean_absolute_error(y_t, y_p),
        "Accuracy": accuracy_score(y_t > 0, cfs),
        "F1": f1_score(y_t > 0, cfs, zero_division=0),
        "AUC": roc_auc_score(y_t > 0, y_p),
    }


#
# 8. MAIN ENTRYPOINT
#
def main():
    # 8.1 Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #
    # 8.2. LOAD DAILY DATA (with forward‐fill on NaNs)
    #
    data, tickers, features, df_day = load_daily_tensor("nasdaq100_5y.csv")
    N, T_days, Fcnt = data.shape
    print(f"Loaded daily data: N={N}, T_days={T_days}, Features={Fcnt}")

    #
    # 8.3. AGGREGATE WEEKLY (COARSE) from the filled daily DataFrame
    #
    df_week = aggregate_weekly(df_day)
    T_weeks = len(df_week)
    print(f"Weekly‐aggregated data: T_weeks={T_weeks}")

    #
    # 8.4. SLIDING WINDOWS ON DAILY DATA FOR FINE‐GRAINED ENCODER
    #      window length = 5 trading days, stride = 5 (non‐overlapping weekly windows)
    #
    fine_L = 5
    stride = 5
    fm, starts = sliding_windows(data, fine_L, stride)  # (N, W, 5*Fcnt)
    W = fm.size(1)
    print(f"Constructed fine‐grained windows: W = {W}")

    # 8.5. EXTRACT COARSE‐GRAINED FEATURES (weekly) AND ALIGN LENGTH
    cm_all = torch.tensor(
        df_week.values.reshape(T_weeks, N, Fcnt).transpose(1, 0, 2), dtype=torch.float
    )  # (N, T_weeks, Fcnt)
    cm = cm_all[:, :W, :]  # truncate to match W if needed
    assert cm.shape[1] == fm.shape[1], "Mismatch between fine windows and coarse windows"

    #
    # 8.6. COMPUTE WEEKLY RETURNS FROM CLOSE PRICES (labels)
    #
    # 8.6.1. Find the “Close” feature
    close_feature = None
    for f in features:
        if f.lower() == "close":
            close_feature = f
            break
    if close_feature is None:
        for f in features:
            if "close" in f.lower():
                close_feature = f
                break
    if close_feature is None:
        raise ValueError(f"No close feature found in features={features}")

    # 8.6.2. Extract weekly close prices: shape = (T_weeks, N)
    close_np = df_week.xs(close_feature, level="feature", axis=1).values

    # 8.6.3. Compute relative return: (P_{t+1} / P_t - 1), transpose → (N, T_weeks-1)
    rel_ret = (close_np[1:] / close_np[:-1] - 1.0).T  # shape (N, T_weeks-1)
    weekly_ret = torch.tensor(rel_ret, dtype=torch.float)

    # 8.6.4. Truncate to W columns so that labels align with fm and cm
    weekly_ret = weekly_ret[:, :W]  # (N, W)

    #
    # 8.7. SPLIT INTO TRAIN / VAL / TEST (60% / 20% / 20% of W)
    #
    tr = int(W * 0.6)
    va = int(W * 0.8)
    fm_tr, fm_va, fm_te = fm[:, :tr], fm[:, tr:va], fm[:, va:]
    cm_tr, cm_va, cm_te = cm[:, :tr], cm[:, tr:va], cm[:, va:]
    lb_tr, lb_va, lb_te = (
        weekly_ret[:, :tr],
        weekly_ret[:, tr:va],
        weekly_ret[:, va:],
    )
    print(
        f"Split W={W} into train={fm_tr.size(1)} / val={fm_va.size(1)} / test={fm_te.size(1)} windows"
    )

    #
    # 8.8. INITIALIZE MODELS AND DUMMY MEMORY MATRICES
    #
    D = 128
    enc_f = FineEncoder(fm_tr.shape[2], D).to(device)  # input_dim = 5 * Fcnt
    enc_c = CoarseEncoder(cm_tr.shape[2], D).to(device)  # input_dim = Fcnt
    reg = Regressor(D).to(device)

    # Dummy memory matrices of shape (M_size, D)
    M_size = 128
    M_f = F.normalize(torch.randn(M_size, D, device=device), dim=1)
    M_c = F.normalize(torch.randn(M_size, D, device=device), dim=1)

    #
    # 8.9. PRE‐TRAINING STAGE
    #
    M_f, M_c = pretrain_stage(
        enc_f, enc_c, fm_tr, cm_tr, M_f, M_c, device=device
    )

    #
    # 8.10. FINE‐TUNING STAGE
    #
    finetune_stage(enc_f, enc_c, reg, fm_tr, cm_tr, lb_tr, device=device)

    #
    # 8.11. EVALUATION ON TEST SET
    #
    metrics = evaluate(reg, enc_c, fm_te, cm_te, lb_te, device=device)
    print("\n=== Evaluation on Test Set ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    #
    # 8.12. BACKTEST / SIMULATION (“Top K-Drop” STRATEGY)
    #
    fm_test, cm_test, lb_test = fm_te, cm_te, lb_te
    Wt = fm_test.size(1)
    preds = []
    rets = []
    for t in range(Wt):
        x_c = cm_test[:, t, :].to(device)
        yhat = reg(enc_c(x_c)).detach().cpu().numpy()  # (N,)
        ytrue = lb_test[:, t].numpy()
        preds.append(yhat)
        rets.append(ytrue)
    preds = np.stack(preds, axis=1)  # (N, Wt)
    rets = np.stack(rets, axis=1)    # (N, Wt)

    topk = 15
    dropk = 5
    capital = 1.0
    equity = [capital]

    for i in range(Wt - 1):
        idx_sort = np.argsort(-preds[:, i])
        top_idx = idx_sort[:topk]
        chosen = top_idx[:-dropk]
        realized = np.mean(rets[chosen, i + 1])
        capital *= 1 + realized
        equity.append(capital)

    #
    # 8.13. PLOTTING: TRUE VS PREDICTED VS EQUITY CURVE
    #
    plt.figure(figsize=(10, 6))

    mean_weekly_ret = rets.mean(axis=0)  # (Wt,)
    cum_true = np.cumprod(1 + mean_weekly_ret)

    # For predicted, we show a normalized cumulative sum (just for illustration)
    cum_pred_raw = np.cumsum(preds.mean(axis=0))
    # Normalize cum_pred_raw to [0,1], then shift to [1,1.5] for visibility
    cum_pred = (cum_pred_raw - cum_pred_raw.min()) / (cum_pred_raw.max() - cum_pred_raw.min())
    cum_pred = 1.0 + 0.5 * cum_pred

    plt.plot(cum_true, label="True Avg CumReturn", linewidth=2)
    plt.plot(cum_pred, label="Pred Avg CumScore", linestyle="--", linewidth=2)
    plt.plot(equity, label="Equity Curve (TopK-Drop)", linestyle=":", linewidth=2)

    plt.title("True vs. Predicted vs. Equity Curve")
    plt.xlabel("Week Index (Test Period)")
    plt.ylabel("Normalized Value / Capital")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("predict.png")
    print("Saved figure → predict.png")


if __name__ == "__main__":
    main()
