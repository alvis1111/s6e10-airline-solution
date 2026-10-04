"""S6E10 — 分组错误分析: 各段 OOF AUC, 找模型弱点所在."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA = "data/playground-series-s6e10"
TARGET = "satisfaction"
N_FOLDS = 5
SEED = 42

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
    for c in TE_COLS: df[c] = df[c].astype("category")

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

oof = np.zeros(len(train))
for tr_idx, va_idx in skf.split(train, y):
    X_tr, X_va = train.iloc[tr_idx], train.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]
    te_tr, te_va, _ = te_encode(X_tr, X_va, test, y_tr, TE_COLS)
    Xt = pd.concat([X_tr[NUM_COLS].reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1)
    Xv = pd.concat([X_va[NUM_COLS].reset_index(drop=True), te_va.reset_index(drop=True)], axis=1)
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(Xt, y_tr, eval_X=Xv, eval_y=y_va, callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[va_idx] = m.predict_proba(Xv)[:, 1]

print(f"整体 OOF AUC = {roc_auc_score(y, oof):.5f}\n", flush=True)

def seg_auc(mask, name):
    if mask.sum() < 50: return
    a = roc_auc_score(y[mask], oof[mask])
    rate = y[mask].mean()
    print(f"{name:32s} n={mask.sum():7d}  正样本率={rate:.3f}  OOF AUC={a:.5f}", flush=True)

print("=== 出行类型 ===", flush=True)
seg_auc(train["Type of Travel"]=="Business travel", "Business travel")
seg_auc(train["Type of Travel"]=="Personal Travel", "Personal Travel")
print("=== 客户类型 ===", flush=True)
seg_auc(train["Customer Type"]=="Loyal Customer", "Loyal")
seg_auc(train["Customer Type"]=="disloyal Customer", "disloyal")
print("=== 舱位 ===", flush=True)
for c in ["Business", "Eco", "Eco Plus"]:
    seg_auc(train["Class"]==c, f"Class={c}")
print("=== Travel × Customer ===", flush=True)
for combo in ["Business travel_Loyal Customer", "Business travel_disloyal Customer",
              "Personal Travel_Loyal Customer", "Personal Travel_disloyal Customer"]:
    seg_auc(train["travel_x_customer"]==combo, combo)
