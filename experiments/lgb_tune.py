"""S6E10 — LGBM 调参 + TE vs 原生类别 A/B 实验 (5折, 只报 OOF)."""
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

def build_features(df):
    df = df.copy()
    df["service_sum"] = df[SERVICE_COLS].sum(axis=1)
    df["service_mean"] = df[SERVICE_COLS].mean(axis=1)
    df["service_std"] = df[SERVICE_COLS].std(axis=1)
    df["total_delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    df["is_delayed"] = (df["total_delay"] > 0).astype(int)
    df["travel_x_class"] = df["Type of Travel"] + "_" + df["Class"]
    df["travel_x_customer"] = df["Type of Travel"] + "_" + df["Customer Type"]
    df["dist_bucket"] = pd.cut(
        df["Flight Distance"], bins=[0, 500, 1000, 2000, 3000, 5000, np.inf],
        labels=False,
    ).astype("category")
    return df

train = build_features(train)
test = build_features(test)

NUM_COLS = [
    "Age", "Flight Distance", "Departure Delay in Minutes",
    "Arrival Delay in Minutes", "service_sum", "service_mean",
    "service_std", "total_delay", "is_delayed",
] + SERVICE_COLS
TE_COLS = CAT_COLS + ["travel_x_class", "travel_x_customer", "dist_bucket"]

for df in (train, test):
    for c in NUM_COLS:
        if df[c].isna().any():
            df[c] = df[c].fillna(0)

def te_encode(X_tr, X_va, X_te, y_tr, cols):
    prior = y_tr.mean()
    out_tr = X_tr[cols].astype(str).copy()
    out_va = X_va[cols].astype(str).copy()
    out_te = X_te[cols].astype(str).copy()
    for c in cols:
        stats = pd.DataFrame({"v": X_tr[c].astype(str), "y": y_tr}).groupby("v")["y"].agg(["mean", "count"])
        enc = (stats["mean"] * stats["count"] + prior * 20.0) / (stats["count"] + 20.0)
        out_tr[c] = out_tr[c].map(enc).fillna(prior)
        out_va[c] = out_va[c].map(enc).fillna(prior)
        out_te[c] = out_te[c].map(enc).fillna(prior)
    return out_tr.astype("float32"), out_va.astype("float32"), out_te.astype("float32")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def run(name, params, use_te=True, use_native_cat=False):
    oof = np.zeros(len(train))
    for tr_idx, va_idx in skf.split(train, y):
        X_tr, X_va = train.iloc[tr_idx], train.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        parts_tr, parts_va = [X_tr[NUM_COLS].reset_index(drop=True)], [X_va[NUM_COLS].reset_index(drop=True)]
        if use_te:
            te_tr, te_va, _ = te_encode(X_tr, X_va, test, y_tr, TE_COLS)
            parts_tr.append(te_tr.reset_index(drop=True))
            parts_va.append(te_va.reset_index(drop=True))
        if use_native_cat:
            cat_tr = X_tr[TE_COLS].astype("category").reset_index(drop=True)
            cat_va = X_va[TE_COLS].astype("category").reset_index(drop=True)
            parts_tr.append(cat_tr)
            parts_va.append(cat_va)
        Xt = pd.concat(parts_tr, axis=1)
        Xv = pd.concat(parts_va, axis=1)
        m = lgb.LGBMClassifier(**params)
        m.fit(Xt, y_tr, eval_X=Xv, eval_y=y_va,
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va_idx] = m.predict_proba(Xv)[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"  {name:40s} OOF = {auc:.5f}", flush=True)
    return auc

base = dict(objective="binary", metric="auc", n_estimators=8000,
            random_state=SEED, n_jobs=-1, verbosity=-1)

configs = [
    # (name, params, use_te, use_native_cat)
    ("TE lr.03 leaves127 (当前)", dict(**base, learning_rate=0.03, num_leaves=127, min_child_samples=50, colsample_bytree=0.7, subsample=0.8, reg_alpha=0.3, reg_lambda=1.0), True, False),
    ("TE lr.02 leaves255", dict(**base, learning_rate=0.02, num_leaves=255, min_child_samples=50, colsample_bytree=0.7, subsample=0.8, reg_alpha=0.3, reg_lambda=1.0), True, False),
    ("TE lr.05 leaves63 col.9", dict(**base, learning_rate=0.05, num_leaves=63, min_child_samples=100, colsample_bytree=0.9, subsample=0.8, reg_alpha=0.1, reg_lambda=1.0), True, False),
    ("原生类别 lr.03 leaves127", dict(**base, learning_rate=0.03, num_leaves=127, min_child_samples=50, colsample_bytree=0.7, subsample=0.8, reg_alpha=0.3, reg_lambda=1.0), False, True),
    ("TE+原生类别 lr.03 leaves127", dict(**base, learning_rate=0.03, num_leaves=127, min_child_samples=50, colsample_bytree=0.7, subsample=0.8, reg_alpha=0.3, reg_lambda=1.0), True, True),
]

print("=== LGBM 调参实验 (5折) ===", flush=True)
results = {}
for name, params, use_te, use_nc in configs:
    results[name] = run(name, params, use_te, use_nc)

print("\n=== 汇总 ===", flush=True)
for k, v in sorted(results.items(), key=lambda x: -x[1]):
    print(f"  {v:.5f}  {k}", flush=True)
