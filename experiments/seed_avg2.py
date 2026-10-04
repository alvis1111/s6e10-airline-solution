"""S6E10 — 补跑 XGBoost(修早停+调参) + CatBoost(快参), 与已缓存的 LGBM 一起 blend."""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
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

def train_xgb(seed):
    params = dict(
        objective="binary:logistic", eval_metric="auc", learning_rate=0.03,
        max_depth=8, min_child_weight=20, colsample_bytree=0.8, subsample=0.9,
        reg_alpha=0.1, reg_lambda=1.0, n_estimators=8000, tree_method="hist",
        random_state=seed, n_jobs=-1, early_stopping_rounds=150,
    )
    oof = np.zeros(len(train)); pred = np.zeros(len(test))
    for tr_idx, va_idx in skf.split(train, y):
        X_tr, X_va = train.iloc[tr_idx], train.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        te_tr, te_va, te_te = te_encode(X_tr, X_va, test, y_tr, TE_COLS)
        Xt = pd.concat([X_tr[NUM_COLS].reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1)
        Xv = pd.concat([X_va[NUM_COLS].reset_index(drop=True), te_va.reset_index(drop=True)], axis=1)
        Xte = pd.concat([test[NUM_COLS].reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)
        m = xgb.XGBClassifier(**params)
        m.fit(Xt, y_tr, eval_set=[(Xv, y_va)], verbose=False)
        oof[va_idx] = m.predict_proba(Xv)[:, 1]
        pred += m.predict_proba(Xte)[:, 1] / N_FOLDS
    np.save(f"{OUT}/oof_xgb_{seed}.npy", oof)
    np.save(f"{OUT}/pred_xgb_{seed}.npy", pred)
    print(f"  [xgb seed={seed}] OOF AUC = {roc_auc_score(y, oof):.5f}", flush=True)

def train_cb(seed):
    params = dict(
        loss_function="Logloss", eval_metric="AUC", learning_rate=0.1,
        depth=6, l2_leaf_reg=3, iterations=3000, random_seed=seed,
        od_type="Iter", od_wait=100, one_hot_max_size=10,
        boosting_type="Plain", verbose=False, thread_count=-1,
    )
    oof = np.zeros(len(train)); pred = np.zeros(len(test))
    for tr_idx, va_idx in skf.split(train, y):
        X_tr, X_va = train.iloc[tr_idx], train.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        Xt = pd.concat([X_tr[NUM_COLS].reset_index(drop=True),
                        X_tr[CAT_COLS].astype(str).reset_index(drop=True)], axis=1)
        Xv = pd.concat([X_va[NUM_COLS].reset_index(drop=True),
                        X_va[CAT_COLS].astype(str).reset_index(drop=True)], axis=1)
        Xte = pd.concat([test[NUM_COLS].reset_index(drop=True),
                         test[CAT_COLS].astype(str).reset_index(drop=True)], axis=1)
        cat_idx = [Xt.columns.get_loc(c) for c in CAT_COLS]
        m = CatBoostClassifier(**params)
        m.fit(Xt, y_tr, eval_set=(Xv, y_va), cat_features=cat_idx, use_best_model=True)
        oof[va_idx] = m.predict_proba(Xv)[:, 1]
        pred += m.predict_proba(Xte)[:, 1] / N_FOLDS
    np.save(f"{OUT}/oof_cb_{seed}.npy", oof)
    np.save(f"{OUT}/pred_cb_{seed}.npy", pred)
    print(f"  [cb seed={seed}] OOF AUC = {roc_auc_score(y, oof):.5f}", flush=True)

print("=== XGBoost (早停修复) ===", flush=True)
for s in SEEDS:
    train_xgb(s)
print("=== CatBoost (快参) ===", flush=True)
for s in SEEDS:
    train_cb(s)

# ---- 汇总 blend: 所有缓存模型 ----
print("\n=== Blend ===", flush=True)
oofs, preds = [], []
for s in SEEDS:
    for name in ["lgb", "xgb", "cb"]:
        f = f"{OUT}/oof_{name}_{s}.npy"
        if os.path.exists(f):
            oofs.append(np.load(f))
            preds.append(np.load(f"{OUT}/pred_{name}_{s}.npy"))
print(f"共 {len(oofs)} 个模型参与 blend", flush=True)
blend_oof = np.mean([rankdata(o) for o in oofs], axis=0) / len(train)
blend_pred = np.mean([rankdata(p) for p in preds], axis=0) / len(test)
print(f"Rank-avg blend OOF AUC = {roc_auc_score(y, blend_oof):.5f}", flush=True)

sub[TARGET] = blend_pred
sub.to_csv("submission_seedavg.csv", index=False)
print("submission saved -> submission_seedavg.csv", flush=True)
