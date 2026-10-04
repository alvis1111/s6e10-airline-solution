"""S6E10 — Permutation Importance: 看哪些特征在真正驱动预测."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

DATA = "data/playground-series-s6e10"
TARGET = "satisfaction"
SEED = 42

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train[TARGET].astype(int).values

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
DIGITAL = ["Inflight wifi service", "Ease of Online booking", "Online boarding"]
COMFORT = ["Food and drink", "Seat comfort", "Inflight entertainment", "Cleanliness"]
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
    df["dist_bucket"] = pd.cut(df["Flight Distance"], bins=[0, 500, 1000, 2000, 3000, 5000, np.inf], labels=False)
    df["digital_mean"] = df[DIGITAL].mean(axis=1)
    df["comfort_mean"] = df[COMFORT].mean(axis=1)
    return df

train = build(train)
test = build(test)

NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
            "service_sum", "service_mean", "service_std", "total_delay", "is_delayed",
            "digital_mean", "comfort_mean"] + SERVICE_COLS
TE_COLS = CAT_COLS + ["travel_x_class", "travel_x_customer", "dist_bucket"]

for df in (train, test):
    for c in NUM_COLS:
        if df[c].isna().any():
            df[c] = df[c].fillna(0)

# 80/20 划分, TE 在训练部分 fit
X_tr, X_val, y_tr, y_val = train_test_split(train, y, test_size=0.2, stratify=y, random_state=SEED)

prior = y_tr.mean()
tr_te = X_tr[TE_COLS].astype(str).copy()
va_te = X_val[TE_COLS].astype(str).copy()
for c in TE_COLS:
    stats = pd.DataFrame({"v": X_tr[c].astype(str), "y": y_tr}).groupby("v")["y"].agg(["mean", "count"])
    enc = (stats["mean"] * stats["count"] + prior * 20.0) / (stats["count"] + 20.0)
    tr_te[c] = X_tr[c].astype(str).map(enc).fillna(prior)
    va_te[c] = X_val[c].astype(str).map(enc).fillna(prior)

Xtr = pd.concat([X_tr[NUM_COLS].reset_index(drop=True), tr_te.reset_index(drop=True)], axis=1)
Xva = pd.concat([X_val[NUM_COLS].reset_index(drop=True), va_te.reset_index(drop=True)], axis=1)

m = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
                       min_child_samples=50, colsample_bytree=0.7, subsample=0.8,
                       reg_alpha=0.3, reg_lambda=1.0, n_estimators=8000,
                       random_state=SEED, n_jobs=-1, verbosity=-1)
m.fit(Xtr, y_tr, eval_X=Xva, eval_y=y_val, callbacks=[lgb.early_stopping(150, verbose=False)])
print(f"验证集 AUC = {roc_auc_score(y_val, m.predict_proba(Xva)[:, 1]):.5f}", flush=True)

# 用子集加速 permutation (3万行)
idx = np.random.RandomState(SEED).choice(len(Xva), 30000, replace=False)
Xsub, ysub = Xva.iloc[idx], y_val[idx]

r = permutation_importance(m, Xsub, ysub, scoring="roc_auc", n_repeats=5,
                           random_state=SEED, n_jobs=-1)
imp = pd.Series(r.importances_mean, index=Xva.columns).sort_values(ascending=False)
print("\n=== Permutation Importance (AUC 下降越多越重要) ===", flush=True)
for name, v in imp.head(35).items():
    bar = "#" * int(v * 800)
    print(f"{name:28s} {v:.5f}  {bar}", flush=True)
