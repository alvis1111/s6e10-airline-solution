"""S6E10 — NN + 目标编码特征: 把 GBDT 的强 TE 特征(尤其 travel_x_customer)当数值喂给 MLP."""
import os, time
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
SEEDS = [42, 202]
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
    df["travel_x_class_x_customer"] = df["Type of Travel"] + "_" + df["Class"] + "_" + df["Customer Type"]
    df["dist_bucket"] = pd.cut(df["Flight Distance"], bins=[0,500,1000,2000,3000,5000,np.inf], labels=False)
    return df

train = build(train); test = build(test)
NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
            "service_sum", "service_mean", "service_std", "total_delay", "is_delayed"] + SERVICE_COLS
TE_COLS = CAT_COLS + ["travel_x_class", "travel_x_customer", "travel_x_class_x_customer", "dist_bucket"]
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

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=(512,256,128), dropout=0.3):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev,1))
        self.mlp = nn.Sequential(*layers)
    def forward(self, x): return self.mlp(x).squeeze(1)

@torch.no_grad()
def predict(model, X, bs=65536):
    model.eval()
    outs = [torch.sigmoid(model(X[i:i+bs])).cpu().numpy() for i in range(0, len(X), bs)]
    return np.concatenate(outs)

def train_fold(X_tr, y_tr, X_va, y_va, seed):
    seed_everything(seed)
    # 标准化在折内 fit
    sc = StandardScaler().fit(X_tr)
    Xt = torch.tensor(sc.transform(X_tr).astype("float32"), device=device)
    Xv = torch.tensor(sc.transform(X_va).astype("float32"), device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    model = MLP(Xt.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(Xt); B = 16384
    best_auc, best_state, patience = 0.0, None, 0
    for epoch in range(60):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, B):
            idx = perm[i:i+B]
            opt.zero_grad(); loss = loss_fn(model(Xt[idx]), yt[idx]); loss.backward(); opt.step()
        sched.step()
        pv = predict(model, Xv)
        auc = roc_auc_score(y_va, pv)
        if auc > best_auc:
            best_auc, best_state, patience = auc, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 10: break
    model.load_state_dict(best_state)
    return predict(model, Xv), best_auc

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
for seed in SEEDS:
    print(f"\n=== NN+TE seed {seed} ===", flush=True)
    oof = np.zeros(len(train)); pred = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        X_tr, X_va = train.iloc[tr_idx], train.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        te_tr, te_va, te_te = te_encode(X_tr, X_va, test, y_tr, TE_COLS)
        Xtr = pd.concat([X_tr[NUM_COLS].reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1).astype("float32")
        Xva = pd.concat([X_va[NUM_COLS].reset_index(drop=True), te_va.reset_index(drop=True)], axis=1).astype("float32")
        Xte = pd.concat([test[NUM_COLS].reset_index(drop=True), te_te.reset_index(drop=True)], axis=1).astype("float32")
        pv, ba = train_fold(Xtr.values, y_tr, Xva.values, y_va, seed)
        oof[va_idx] = pv
        # 测试预测: 用本折 scaler + 模型
        sc = StandardScaler().fit(Xtr.values)
        model_pred = None  # 简化: 每折重训一个模型做测试预测
        # 重训 (用最终 best_state 已在 train_fold 内, 这里重新训一次取测试预测)
        seed_everything(seed)
        Xt = torch.tensor(sc.transform(Xtr.values).astype("float32"), device=device)
        Xv_t = torch.tensor(sc.transform(Xva.values).astype("float32"), device=device)
        yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
        m2 = MLP(Xt.shape[1]).to(device)
        opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3, weight_decay=1e-5)
        best_auc2, best_state2, pat2 = 0.0, None, 0
        for epoch in range(60):
            m2.train(); perm = torch.randperm(len(Xt), device=device)
            for i in range(0, len(Xt), 16384):
                idx = perm[i:i+16384]
                opt2.zero_grad(); loss = nn.BCEWithLogitsLoss()(m2(Xt[idx]), yt[idx]); loss.backward(); opt2.step()
            pv2 = predict(m2, Xv_t); a2 = roc_auc_score(y_va, pv2)
            if a2 > best_auc2: best_auc2, best_state2, pat2 = a2, {k:v.clone() for k,v in m2.state_dict().items()}, 0
            else:
                pat2 += 1
                if pat2 >= 10: break
        m2.load_state_dict(best_state2)
        Xte_t = torch.tensor(sc.transform(Xte.values).astype("float32"), device=device)
        pred += predict(m2, Xte_t) / N_FOLDS
        print(f"  fold {fold}: val_auc={ba:.5f}", flush=True)
    auc = roc_auc_score(y, oof)
    np.save(f"{OUT}/oof_nnte_{seed}.npy", oof)
    np.save(f"{OUT}/pred_nnte_{seed}.npy", pred)
    print(f"  NN+TE seed {seed} OOF AUC = {auc:.5f}", flush=True)
