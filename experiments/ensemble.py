"""S6E10 — 3模型集成: LGBM(TE) + XGBoost(TE) + CatBoost(原生类别).

沿用特征工程, 三模型各自 5 折 OOF, 最后 rank-average blend.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

DATA = "data/playground-series-s6e10"
TARGET = "satisfaction"
N_FOLDS = 5
SEED = 42

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
sub = pd.read_csv(f"{DATA}/sample_submission.csv")
y = train[TARGET].astype(int)

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

def smoothed_te(tr, va, y_tr, prior, m=20.0):
    stats = pd.DataFrame({"v": tr, "y": y_tr}).groupby("v")["y"].agg(["mean", "count"])
    enc = (stats["mean"] * stats["count"] + prior * m) / (stats["count"] + m)
    return tr.map(enc), va.map(enc)

def te_encode_all(X_tr, X_va, X_te, y_tr):
    prior = y_tr.mean()
    tr_te = X_tr[TE_COLS].copy()
    va_te = X_va[TE_COLS].copy()
    te_te = X_te[TE_COLS].copy()
    for c in TE_COLS:
        tr_enc, va_enc = smoothed_te(X_tr[c], X_va[c], y_tr, prior)
        tr_te[c] = tr_enc.astype("float32")
        va_te[c] = va_enc.astype("float32")
        stats = pd.DataFrame({"v": X_tr[c], "y": y_tr}).groupby("v")["y"].agg(["mean", "count"])
        te_map = (stats["mean"] * stats["count"] + prior * 20.0) / (stats["count"] + 20.0)
        te_te[c] = X_te[c].map(te_map).astype("float32").fillna(prior)
    return (
        pd.concat([X_tr[NUM_COLS].reset_index(drop=True), tr_te.reset_index(drop=True)], axis=1),
        pd.concat([X_va[NUM_COLS].reset_index(drop=True), va_te.reset_index(drop=True)], axis=1),
        pd.concat([X_te[NUM_COLS].reset_index(drop=True), te_te.reset_index(drop=True)], axis=1),
    )

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# 每模型存一份 OOF 和 test pred, 最后 rank-average
oof_dict = {}
pred_dict = {}

# ---- LightGBM (TE) ----
lgb_params = dict(
    objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
    min_child_samples=50, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
    reg_alpha=0.3, reg_lambda=1.0, n_estimators=8000, random_state=SEED,
    n_jobs=-1, verbosity=-1,
)
oof_lgb = np.zeros(len(train)); pred_lgb = np.zeros(len(test))
for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
    X_tr, X_va = train.iloc[tr_idx], train.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    Xt, Xv, Xte = te_encode_all(X_tr, X_va, test, y_tr)
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(Xt, y_tr, eval_X=Xv, eval_y=y_va, callbacks=[lgb.early_stopping(150, verbose=False)])
    oof_lgb[va_idx] = m.predict_proba(Xv)[:, 1]
    pred_lgb += m.predict_proba(Xte)[:, 1] / N_FOLDS
oof_dict["lgb"] = oof_lgb; pred_dict["lgb"] = pred_lgb
print(f"LGBM   OOF AUC = {roc_auc_score(y, oof_lgb):.5f}")

# ---- XGBoost (TE) ----
xgb_params = dict(
    objective="binary:logistic", eval_metric="auc", learning_rate=0.03,
    max_depth=7, min_child_weight=40, colsample_bytree=0.7, subsample=0.8,
    reg_alpha=0.3, reg_lambda=1.0, n_estimators=8000, tree_method="hist",
    random_state=SEED, n_jobs=-1,
)
oof_xgb = np.zeros(len(train)); pred_xgb = np.zeros(len(test))
for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
    X_tr, X_va = train.iloc[tr_idx], train.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    Xt, Xv, Xte = te_encode_all(X_tr, X_va, test, y_tr)
    m = xgb.XGBClassifier(**xgb_params)
    m.fit(Xt, y_tr, eval_set=[(Xv, y_va)], verbose=False)
    oof_xgb[va_idx] = m.predict_proba(Xv)[:, 1]
    pred_xgb += m.predict_proba(Xte)[:, 1] / N_FOLDS
oof_dict["xgb"] = oof_xgb; pred_dict["xgb"] = pred_xgb
print(f"XGBoost OOF AUC = {roc_auc_score(y, oof_xgb):.5f}")

# ---- CatBoost (原生类别) ----
# CatBoost 直接吃原始类别列, 不用 TE; 数值列 + 原始类别
cb_feat_cols = NUM_COLS + CAT_COLS
cb_params = dict(
    loss_function="Logloss", eval_metric="AUC", learning_rate=0.05,
    depth=8, l2_leaf_reg=5, iterations=8000, random_seed=SEED,
    od_type="Iter", od_wait=150, verbose=False, thread_count=-1,
)
oof_cb = np.zeros(len(train)); pred_cb = np.zeros(len(test))
for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
    X_tr, X_va = train.iloc[tr_idx], train.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    Xt = pd.concat([X_tr[NUM_COLS].reset_index(drop=True),
                    X_tr[CAT_COLS].reset_index(drop=True)], axis=1)
    Xv = pd.concat([X_va[NUM_COLS].reset_index(drop=True),
                    X_va[CAT_COLS].reset_index(drop=True)], axis=1)
    Xte = pd.concat([test[NUM_COLS].reset_index(drop=True),
                     test[CAT_COLS].reset_index(drop=True)], axis=1)
    cat_idx = [Xt.columns.get_loc(c) for c in CAT_COLS]
    m = CatBoostClassifier(**cb_params)
    m.fit(Xt, y_tr, eval_set=(Xv, y_va), cat_features=cat_idx, use_best_model=True)
    oof_cb[va_idx] = m.predict_proba(Xv)[:, 1]
    pred_cb += m.predict_proba(Xte)[:, 1] / N_FOLDS
oof_dict["cb"] = oof_cb; pred_dict["cb"] = pred_cb
print(f"CatBoost OOF AUC = {roc_auc_score(y, oof_cb):.5f}")

# ---- rank-average blend ----
blend_oof = sum(rankdata(v) for v in oof_dict.values()) / len(oof_dict)
blend_pred = sum(rankdata(v) for v in pred_dict.values()) / len(pred_dict)
blend_oof = blend_oof / len(train)
blend_pred = blend_pred / len(test)
print(f"\nBlend (rank-avg) OOF AUC = {roc_auc_score(y, blend_oof):.5f}")

# 也试简单加权平均
from itertools import product
best = (0, None)
for w1 in np.arange(0, 1.01, 0.1):
    for w2 in np.arange(0, 1.01 - w1, 0.1):
        w3 = 1 - w1 - w2
        wavg = w1 * oof_lgb + w2 * oof_xgb + w3 * oof_cb
        a = roc_auc_score(y, wavg)
        if a > best[0]:
            best = (a, (round(w1,1), round(w2,1), round(w3,1)))
print(f"Best weighted avg OOF AUC = {best[0]:.5f}  weights(lgb,xgb,cb)={best[1]}")

sub[TARGET] = blend_pred
sub.to_csv("submission_blend.csv", index=False)
print("submission saved -> submission_blend.csv")
