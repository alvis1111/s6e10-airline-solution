"""S6E10 — 原始真实数据辅助建模: 合并训练 + 加权, 看能否突破 0.959."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA = "data/playground-series-s6e10"
REAL = r"C:\Users\GEM07\.cache\kagglehub\datasets\teejmahal20\airline-passenger-satisfaction\versions\1"
TARGET = "satisfaction"
N_FOLDS = 5
SEED = 42

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

syn_tr = build(pd.read_csv(f"{DATA}/train.csv"))
syn_te = build(pd.read_csv(f"{DATA}/test.csv"))
y_syn = syn_tr[TARGET].astype(int).values

real_tr = pd.read_csv(f"{REAL}/train.csv")
real_tr[TARGET] = (real_tr[TARGET] == "satisfied").astype(int)
real_tr = real_tr.drop(columns=["Unnamed: 0", "id", "Inflight service"])
real_tr = build(real_tr)
y_real = real_tr[TARGET].astype(int).values
print(f"真实 train {real_tr.shape} 正样本率 {y_real.mean():.4f} | 合成 train {syn_tr.shape} 正样本率 {y_syn.mean():.4f}", flush=True)

NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
            "service_sum", "service_mean", "service_std", "total_delay", "is_delayed"] + SERVICE_COLS
TE_COLS = CAT_COLS + ["travel_x_class", "travel_x_customer", "dist_bucket"]

for df in (syn_tr, syn_te, real_tr):
    for c in NUM_COLS:
        if df[c].isna().any(): df[c] = df[c].fillna(0)
    for c in TE_COLS:
        df[c] = df[c].astype("category")

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

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
lgb_params = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
                  min_child_samples=50, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                  reg_alpha=0.3, reg_lambda=1.0, n_estimators=8000, random_state=SEED, n_jobs=-1, verbosity=-1)

def run(include_real, real_weight=1.0):
    oof = np.zeros(len(syn_tr))
    for tr_idx, va_idx in skf.split(syn_tr, y_syn):
        X_tr_syn = syn_tr.iloc[tr_idx]; y_tr_syn = y_syn[tr_idx]
        if include_real:
            X_tr = pd.concat([real_tr, X_tr_syn], axis=0, ignore_index=True)
            y_tr = np.concatenate([y_real, y_tr_syn])
            sw = np.concatenate([np.full(len(real_tr), real_weight), np.ones(len(tr_idx))])
        else:
            X_tr = X_tr_syn; y_tr = y_tr_syn; sw = None
        X_va = syn_tr.iloc[va_idx]; y_va = y_syn[va_idx]
        te_tr, te_va, _ = te_encode(X_tr, X_va, syn_te, y_tr, TE_COLS)
        Xt = pd.concat([X_tr[NUM_COLS].reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1)
        Xv = pd.concat([X_va[NUM_COLS].reset_index(drop=True), te_va.reset_index(drop=True)], axis=1)
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(Xt, y_tr, sample_weight=sw, eval_X=Xv, eval_y=y_va, callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va_idx] = m.predict_proba(Xv)[:, 1]
    return roc_auc_score(y_syn, oof)

print("\n=== 真实数据辅助实验 ===", flush=True)
print(f"A. 仅合成 (基线)             OOF = {run(False):.5f}", flush=True)
print(f"B. 合并训练 (等权)           OOF = {run(True, 1.0):.5f}", flush=True)
print(f"C. 合并训练 (真实加权 3x)     OOF = {run(True, 3.0):.5f}", flush=True)
print(f"D. 合并训练 (真实加权 10x)    OOF = {run(True, 10.0):.5f}", flush=True)
