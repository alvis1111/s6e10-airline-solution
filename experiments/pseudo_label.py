"""S6E10 — 伪标签: 用 AutoGluon 测试集预测当软标签, 加回训练集再训 LGBM."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

DATA = "data/playground-series-s6e10"
TARGET = "satisfaction"
SEEDS = [42, 202, 7]

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

# 伪标签: AutoGluon 预测 -> 二值化 (硬伪标签)
pseudo = pd.read_csv("submission_autogluon.csv")
pseudo = pseudo.sort_values("id")[TARGET].values.astype(float)
pseudo_bin = (pseudo >= 0.5).astype(int)
test_idx_start = len(train)

# 合并 train + 伪标签 test
full = pd.concat([train, test], axis=0, ignore_index=True)
full_y = np.concatenate([y, pseudo_bin])
n_test = len(test)

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

skf = KFold(n_splits=5, shuffle=True, random_state=42)
test_pred_all = np.zeros(n_test)

lgb_params = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
                  min_child_samples=50, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                  reg_alpha=0.3, reg_lambda=1.0, n_estimators=8000, n_jobs=-1, verbosity=-1)

for seed in SEEDS:
    oof = np.zeros(len(full))
    for tr_idx, va_idx in skf.split(full, full_y):
        X_tr, X_va = full.iloc[tr_idx], full.iloc[va_idx]
        y_tr, y_va = full_y[tr_idx], full_y[va_idx]
        te_tr, te_va, _ = te_encode(X_tr, X_va, test, y_tr, TE_COLS)
        Xt = pd.concat([X_tr[NUM_COLS].reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1)
        Xv = pd.concat([X_va[NUM_COLS].reset_index(drop=True), te_va.reset_index(drop=True)], axis=1)
        m = lgb.LGBMClassifier(**{**lgb_params, "random_state": seed})
        m.fit(Xt, y_tr, eval_X=Xv, eval_y=y_va, callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va_idx] = m.predict_proba(Xv)[:, 1]
    # test 部分的 OOF 就是"精炼后"的 test 预测
    test_pred_all += oof[test_idx_start:] / len(SEEDS)
    train_auc = roc_auc_score(y, oof[:test_idx_start])
    print(f"seed {seed}: train部分 OOF = {train_auc:.5f}", flush=True)

sub = pd.DataFrame({"id": test["id"], TARGET: test_pred_all})
sub.to_csv("submission_pseudo.csv", index=False)
print(f"\nsaved submission_pseudo.csv  pred mean={test_pred_all.mean():.4f}", flush=True)
