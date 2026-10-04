"""S6E10 — TabM: 共享数值 embedding + K 个子网络平均 (参数高效集成)."""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

DATA = "data/playground-series-s6e10"
TARGET = "satisfaction"
N_FOLDS = 5
SEED = 42
OUT = "cache"
os.makedirs(OUT, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device, flush=True)

def seed_everything(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train[TARGET].astype(int).values

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
SERVICE_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]

def build(df):
    df = df.copy()
    df["service_sum"] = df[SERVICE_COLS].sum(axis=1)
    df["service_mean"] = df[SERVICE_COLS].mean(axis=1)
    df["service_std"] = df[SERVICE_COLS].std(axis=1)
    df["total_delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    df["is_delayed"] = (df["total_delay"] > 0).astype(int)
    df["travel_x_class"] = df["Type of Travel"] + "_" + df["Class"]
    df["travel_x_customer"] = df["Type of Travel"] + "_" + df["Customer Type"]
    df["dist_bucket"] = pd.cut(df["Flight Distance"], bins=[0,500,1000,2000,3000,5000,np.inf], labels=False)
    return df

train = build(train); test = build(test)
NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
            "service_sum", "service_mean", "service_std", "total_delay", "is_delayed"] + SERVICE_COLS
TE_COLS = CAT_COLS + ["travel_x_class", "travel_x_customer", "dist_bucket"]
for df in (train, test):
    for c in NUM_COLS:
        if df[c].isna().any(): df[c] = df[c].fillna(0)

def te_encode(X_tr, X_va, X_te, y_tr, cols):
    prior = y_tr.mean()
    out_tr = X_tr[cols].astype(str).copy(); out_va = X_va[cols].astype(str).copy(); out_te = X_te[cols].astype(str).copy()
    for c in cols:
        stats = pd.DataFrame({"v": X_tr[c].astype(str), "y": y_tr}).groupby("v")["y"].agg(["mean","count"])
        enc = (stats["mean"]*stats["count"] + prior*20.0)/(stats["count"]+20.0)
        out_tr[c] = X_tr[c].astype(str).map(enc).fillna(prior)
        out_va[c] = X_va[c].astype(str).map(enc).fillna(prior)
        out_te[c] = X_te[c].astype(str).map(enc).fillna(prior)
    return out_tr.astype("float32"), out_va.astype("float32"), out_te.astype("float32")

class TabM(nn.Module):
    def __init__(self, n_num, d_emb=16, n_blocks=32, width=256, dropout=0.1):
        super().__init__()
        self.num_emb = nn.Linear(1, d_emb)   # 共享
        in_dim = n_num * d_emb
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, 1))
            for _ in range(n_blocks)
        ])
    def forward(self, x):
        e = self.num_emb(x.unsqueeze(-1)).flatten(1)
        outs = torch.stack([b(e) for b in self.blocks], dim=0)  # (K, B, 1)
        return outs.mean(0).squeeze(1)

@torch.no_grad()
def predict(model, X, bs=65536):
    model.eval()
    outs = [torch.sigmoid(model(X[i:i+bs])).cpu().numpy() for i in range(0, len(X), bs)]
    return np.concatenate(outs)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
    X_tr, X_va = train.iloc[tr_idx], train.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    te_tr, te_va, _ = te_encode(X_tr, X_va, test, y_tr, TE_COLS)
    Xtr = pd.concat([X_tr[NUM_COLS].reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1).astype("float32").values
    Xva = pd.concat([X_va[NUM_COLS].reset_index(drop=True), te_va.reset_index(drop=True)], axis=1).astype("float32").values
    sc = StandardScaler().fit(Xtr)
    Xt = torch.tensor(sc.transform(Xtr), device=device)
    Xv = torch.tensor(sc.transform(Xva), device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    seed_everything(SEED)
    model = TabM(Xt.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(Xt); B = 8192
    best_auc, best_state, patience = 0.0, None, 0
    for epoch in range(40):
        model.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, B):
            idx = perm[i:i+B]
            opt.zero_grad(); loss = loss_fn(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
        pv = predict(model, Xv); auc = roc_auc_score(y_va, pv)
        if auc > best_auc:
            best_auc, best_state, patience = auc, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 7: break
    model.load_state_dict(best_state)
    oof[va_idx] = predict(model, Xv)
    print(f"  fold {fold}: val_auc={best_auc:.5f}", flush=True)
print(f"TabM OOF = {roc_auc_score(y, oof):.5f}", flush=True)
np.save(f"{OUT}/oof_tabm_{SEED}.npy", oof)
