"""Kaggle Playground S6E10 — Predicting Airline Satisfaction

最终方案：目标编码 + LGBM/XGBoost/CatBoost 多种子 5 折，rank-average 集成。
复现 Public LB 0.95862 / OOF 0.95914。

用法：python s6e10_final.py   （数据放 data/playground-series-s6e10/）
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

# ---- 配置 ----
DATA = "data/playground-series-s6e10"   # Kaggle 环境: /kaggle/input/playground-series-s6e10
TARGET = "satisfaction"
N_FOLDS = 5
SEEDS = [42, 202, 7]

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
SERVICE_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]

# ---- 读数据 + 特征工程 ----
train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
sub = pd.read_csv(f"{DATA}/sample_submission.csv")
y = train[TARGET].astype(int).values

def build_features(df):
    df = df.copy()
    df["service_sum"] = df[SERVICE_COLS].sum(axis=1)
    df["service_mean"] = df[SERVICE_COLS].mean(axis=1)
    df["service_std"] = df[SERVICE_COLS].std(axis=1)
    df["total_delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    df["is_delayed"] = (df["total_delay"] > 0).astype(int)
    df["travel_x_class"] = df["Type of Travel"] + "_" + df["Class"]
    df["travel_x_customer"] = df["Type of Travel"] + "_" + df["Customer Type"]
    df["dist_bucket"] = pd.cut(df["Flight Distance"], bins=[0, 500, 1000, 2000, 3000, 5000, np.inf], labels=False)
    return df

train = build_features(train)
test = build_features(test)

NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
            "service_sum", "service_mean", "service_std", "total_delay", "is_delayed"] + SERVICE_COLS
TE_COLS = CAT_COLS + ["travel_x_class", "travel_x_customer", "dist_bucket"]

for df in (train, test):
    for c in NUM_COLS:
        if df[c].isna().any():
            df[c] = df[c].fillna(0)
    for c in TE_COLS:
        df[c] = df[c].astype("category")

# ---- 目标编码 (fold 内 OOF, 平滑 m=20) ----
def te_encode(X_tr, X_va, X_te, y_tr, cols, m=20.0):
    prior = y_tr.mean()
    out_tr = X_tr[cols].astype(str).copy()
    out_va = X_va[cols].astype(str).copy()
    out_te = X_te[cols].astype(str).copy()
    for c in cols:
        stats = pd.DataFrame({"v": X_tr[c].astype(str), "y": y_tr}).groupby("v")["y"].agg(["mean", "count"])
        enc = (stats["mean"] * stats["count"] + prior * m) / (stats["count"] + m)
        out_tr[c] = out_tr[c].map(enc).fillna(prior)
        out_va[c] = out_va[c].map(enc).fillna(prior)
        out_te[c] = out_te[c].map(enc).fillna(prior)
    return out_tr.astype("float32"), out_va.astype("float32"), out_te.astype("float32")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

def run_gbdt(name, params, seed):
    """5 折训练一个 GBDT (TE 特征), 返回 (oof, test_pred)."""
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    for tr_idx, va_idx in skf.split(train, y):
        X_tr, X_va = train.iloc[tr_idx], train.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        te_tr, te_va, te_te = te_encode(X_tr, X_va, test, y_tr, TE_COLS)
        Xt = pd.concat([X_tr[NUM_COLS].reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1)
        Xv = pd.concat([X_va[NUM_COLS].reset_index(drop=True), te_va.reset_index(drop=True)], axis=1)
        Xte = pd.concat([test[NUM_COLS].reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)

        if name == "lgb":
            m = lgb.LGBMClassifier(**params)
            m.fit(Xt, y_tr, eval_X=Xv, eval_y=y_va, callbacks=[lgb.early_stopping(150, verbose=False)])
        elif name == "xgb":
            m = xgb.XGBClassifier(**params)
            m.fit(Xt, y_tr, eval_set=[(Xv, y_va)], verbose=False)
        elif name == "cb":
            Xt = pd.concat([X_tr[NUM_COLS].reset_index(drop=True), X_tr[CAT_COLS].astype(str).reset_index(drop=True)], axis=1)
            Xv = pd.concat([X_va[NUM_COLS].reset_index(drop=True), X_va[CAT_COLS].astype(str).reset_index(drop=True)], axis=1)
            Xte = pd.concat([test[NUM_COLS].reset_index(drop=True), test[CAT_COLS].astype(str).reset_index(drop=True)], axis=1)
            cat_idx = [Xt.columns.get_loc(c) for c in CAT_COLS]
            m = CatBoostClassifier(**params)
            m.fit(Xt, y_tr, eval_set=(Xv, y_va), cat_features=cat_idx, use_best_model=True)

        oof[va_idx] = m.predict_proba(Xv)[:, 1]
        pred += m.predict_proba(Xte)[:, 1] / N_FOLDS
    return oof, pred

lgb_params = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
                  min_child_samples=50, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                  reg_alpha=0.3, reg_lambda=1.0, n_estimators=8000, n_jobs=-1, verbosity=-1)
xgb_params = dict(objective="binary:logistic", eval_metric="auc", learning_rate=0.03,
                  max_depth=8, min_child_weight=20, colsample_bytree=0.8, subsample=0.9,
                  reg_alpha=0.1, reg_lambda=1.0, n_estimators=8000, tree_method="hist",
                  n_jobs=-1, early_stopping_rounds=150)
cb_params = dict(loss_function="Logloss", eval_metric="AUC", learning_rate=0.1,
                 depth=6, l2_leaf_reg=3, iterations=3000, od_type="Iter", od_wait=100,
                 one_hot_max_size=10, boosting_type="Plain", verbose=False, thread_count=-1)

# ---- 训练 + 集成 ----
oofs, preds = [], []
for s in SEEDS:
    print(f"\n=== seed {s} ===", flush=True)
    o, p = run_gbdt("lgb", {**lgb_params, "random_state": s}, s)
    oofs.append(o); preds.append(p)
    print(f"  LGBM     OOF = {roc_auc_score(y, o):.5f}", flush=True)

    o, p = run_gbdt("xgb", {**xgb_params, "random_state": s}, s)
    oofs.append(o); preds.append(p)
    print(f"  XGBoost  OOF = {roc_auc_score(y, o):.5f}", flush=True)

    o, p = run_gbdt("cb", {**cb_params, "random_seed": s}, s)
    oofs.append(o); preds.append(p)
    print(f"  CatBoost OOF = {roc_auc_score(y, o):.5f}", flush=True)

# ---- rank-average blend ----
blend_oof = np.mean([rankdata(o) for o in oofs], axis=0) / len(train)
blend_pred = np.mean([rankdata(p) for p in preds], axis=0) / len(test)
print(f"\n=== {len(oofs)} 模型 rank-avg blend OOF AUC = {roc_auc_score(y, blend_oof):.5f} ===")

sub[TARGET] = blend_pred
sub.to_csv("submission_final.csv", index=False)
print("saved -> submission_final.csv")
