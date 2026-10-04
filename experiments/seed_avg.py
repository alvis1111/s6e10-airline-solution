"""S6E10 — LGBM + XGB 多种子平均 (纯 GBDT 路线).

每个 (模型, 种子) 跑 5 折, OOF 和 test 预测立即落盘 (npy),
这样中途中断也不丢已完成的模型。最后 rank-average blend.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

DATA = "data/playground-series-s6e10"
TARGET = "satisfaction"
N_FOLDS = 5
SEEDS = [42, 202, 7]
OUT = "cache"
os.makedirs(OUT, exist_ok=True)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
sub = pd.read_csv(f"{DATA}/sample_submission.csv")
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
    for c in TE_COLS:
        df[c] = df[c].astype("category")

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

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

def run_model(name, params, use_te=True, seed=42):
    """训练一个模型 (5折), 落盘 oof/pred. 返回 oof."""
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        X_tr, X_va = train.iloc[tr_idx], train.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        if use_te:
            te_tr, te_va, te_te = te_encode(X_tr, X_va, test, y_tr, TE_COLS)
            Xt = pd.concat([X_tr[NUM_COLS].reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1)
            Xv = pd.concat([X_va[NUM_COLS].reset_index(drop=True), te_va.reset_index(drop=True)], axis=1)
            Xte = pd.concat([test[NUM_COLS].reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)
        else:
            Xt = X_tr[NUM_COLS].reset_index(drop=True)
            Xv = X_va[NUM_COLS].reset_index(drop=True)
            Xte = test[NUM_COLS].reset_index(drop=True)

        if name == "lgb":
            m = lgb.LGBMClassifier(**params)
            m.fit(Xt, y_tr, eval_X=Xv, eval_y=y_va,
                  callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[va_idx] = m.predict_proba(Xv)[:, 1]
            pred += m.predict_proba(Xte)[:, 1] / N_FOLDS
        elif name == "xgb":
            m = xgb.XGBClassifier(**params)
            m.fit(Xt, y_tr, eval_set=[(Xv, y_va)], verbose=False)
            oof[va_idx] = m.predict_proba(Xv)[:, 1]
            pred += m.predict_proba(Xte)[:, 1] / N_FOLDS

    auc = roc_auc_score(y, oof)
    np.save(f"{OUT}/oof_{name}_{seed}.npy", oof)
    np.save(f"{OUT}/pred_{name}_{seed}.npy", pred)
    print(f"  [{name} seed={seed}] OOF AUC = {auc:.5f}", flush=True)
    return auc

lgb_params = dict(
    objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
    min_child_samples=50, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
    reg_alpha=0.3, reg_lambda=1.0, n_estimators=8000,
    random_state=42, n_jobs=-1, verbosity=-1,
)
xgb_params = dict(
    objective="binary:logistic", eval_metric="auc", learning_rate=0.03,
    max_depth=7, min_child_weight=40, colsample_bytree=0.7, subsample=0.8,
    reg_alpha=0.3, reg_lambda=1.0, n_estimators=8000, tree_method="hist",
    random_state=42, n_jobs=-1,
)

print("=== LGBM 多种子 ===")
for s in SEEDS:
    run_model("lgb", {**lgb_params, "random_state": s}, use_te=True, seed=s)

print("=== XGBoost 多种子 ===")
for s in SEEDS:
    run_model("xgb", {**xgb_params, "random_state": s}, use_te=True, seed=s)

# ---- blend ----
print("\n=== Blend ===")
oofs, preds = [], []
for s in SEEDS:
    for name in ["lgb", "xgb"]:
        oofs.append(np.load(f"{OUT}/oof_{name}_{s}.npy"))
        preds.append(np.load(f"{OUT}/pred_{name}_{s}.npy"))

# rank-average
blend_oof = np.mean([rankdata(o) for o in oofs], axis=0) / len(train)
blend_pred = np.mean([rankdata(p) for p in preds], axis=0) / len(test)
print(f"Rank-avg blend OOF AUC = {roc_auc_score(y, blend_oof):.5f}")

# 简单平均
avg_oof = np.mean(oofs, axis=0)
avg_pred = np.mean(preds, axis=0)
print(f"Simple-avg blend OOF AUC = {roc_auc_score(y, avg_oof):.5f}")

sub[TARGET] = blend_pred
sub.to_csv("submission_seedavg.csv", index=False)
print("submission saved -> submission_seedavg.csv")
