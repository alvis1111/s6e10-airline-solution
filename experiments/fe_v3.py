"""S6E10 — 交互目标编码: WiFi×Travel×Customer / OnlineBoarding×Class 等 3-way 组合."""
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
    # 交互: wifi / online boarding 分桶 + 类别组合
    df["wifi_x_travel"] = df["Inflight wifi service"].astype(str) + "_" + df["Type of Travel"]
    df["wifi_x_customer"] = df["Inflight wifi service"].astype(str) + "_" + df["Customer Type"]
    df["wifi_x_travel_x_customer"] = df["Inflight wifi service"].astype(str) + "_" + df["Type of Travel"] + "_" + df["Customer Type"]
    df["ob_x_class"] = df["Online boarding"].astype(str) + "_" + df["Class"]
    df["ob_x_class_x_customer"] = df["Online boarding"].astype(str) + "_" + df["Class"] + "_" + df["Customer Type"]
    return df

train = build(train); test = build(test)
NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
            "service_sum", "service_mean", "service_std", "total_delay", "is_delayed"] + SERVICE_COLS
BASE_TE = CAT_COLS + ["travel_x_class", "travel_x_customer", "dist_bucket"]
NEW_TE = ["wifi_x_travel", "wifi_x_customer", "wifi_x_travel_x_customer", "ob_x_class", "ob_x_class_x_customer"]

for df in (train, test):
    for c in NUM_COLS:
        if df[c].isna().any(): df[c] = df[c].fillna(0)

def te_encode(X_tr, X_va, X_te, y_tr, cols, m=20.0):
    prior = y_tr.mean()
    out_tr = X_tr[cols].astype(str).copy(); out_va = X_va[cols].astype(str).copy(); out_te = X_te[cols].astype(str).copy()
    for c in cols:
        stats = pd.DataFrame({"v": X_tr[c].astype(str), "y": y_tr}).groupby("v")["y"].agg(["mean","count"])
        enc = (stats["mean"]*stats["count"] + prior*m)/(stats["count"]+m)
        out_tr[c] = X_tr[c].astype(str).map(enc).fillna(prior)
        out_va[c] = X_va[c].astype(str).map(enc).fillna(prior)
        out_te[c] = X_te[c].astype(str).map(enc).fillna(prior)
    return out_tr.astype("float32"), out_va.astype("float32"), out_te.astype("float32")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def run(name, te_cols):
    lgb_params = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
                      min_child_samples=50, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                      reg_alpha=0.3, reg_lambda=1.0, n_estimators=8000, random_state=SEED, n_jobs=-1, verbosity=-1)
    oof = np.zeros(len(train))
    for tr_idx, va_idx in skf.split(train, y):
        X_tr, X_va = train.iloc[tr_idx], train.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        te_tr, te_va, _ = te_encode(X_tr, X_va, test, y_tr, te_cols)
        Xt = pd.concat([X_tr[NUM_COLS].reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1)
        Xv = pd.concat([X_va[NUM_COLS].reset_index(drop=True), te_va.reset_index(drop=True)], axis=1)
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(Xt, y_tr, eval_X=Xv, eval_y=y_va, callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va_idx] = m.predict_proba(Xv)[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"  {name:40s} OOF = {auc:.5f}", flush=True)

print("=== 交互目标编码实验 ===", flush=True)
run("A. 基线 TE (2-way)", BASE_TE)
run("B. +3-way 交互 TE", BASE_TE + NEW_TE)
