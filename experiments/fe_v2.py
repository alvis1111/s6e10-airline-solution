"""S6E10 — 洞察驱动特征: 0=不适用(数字类)/0=差(舒适类) 分开 + 分组均值."""
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
# 洞察分组: 数字体验 vs 机上舒适
DIGITAL = ["Inflight wifi service", "Ease of Online booking", "Online boarding"]
COMFORT = ["Food and drink", "Seat comfort", "Inflight entertainment", "Cleanliness"]
SERVICE_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]
REST = [c for c in SERVICE_COLS if c not in DIGITAL + COMFORT]

def build_base(df):
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

def add_insight_feats(df):
    """0=不适用 vs 0=差 分开处理 + 分组均值 + 0感知聚合."""
    df = df.copy()
    # 分组均值
    df["digital_mean"] = df[DIGITAL].mean(axis=1)
    df["comfort_mean"] = df[COMFORT].mean(axis=1)
    df["rest_mean"] = df[REST].mean(axis=1)
    # 每个服务特征的 0 标志 (让 GBDT 自己学方向: 数字类0=好事, 舒适类0=坏事)
    for c in SERVICE_COLS:
        df[f"{c[:12]}_is0"] = (df[c] == 0).astype(int)
    # 数字类 0 个数 (不适用=偏正面) / 舒适类 0 个数 (差=偏负面)
    df["digital_zero_cnt"] = df[[f"{c[:12]}_is0" for c in DIGITAL]].sum(axis=1)
    df["comfort_zero_cnt"] = df[[f"{c[:12]}_is0" for c in COMFORT]].sum(axis=1)
    # 0感知均值: 数字类把0当缺失(不适用), 舒适类0保留(差)
    df["digital_mean_nz"] = df[DIGITAL].replace(0, np.nan).mean(axis=1)
    df["digital_mean_nz"] = df["digital_mean_nz"].fillna(df["digital_mean"])
    # 距离相对同类旅客的偏差 (group statistics)
    df["dist_ratio_class"] = df["Flight Distance"] / df.groupby("Class")["Flight Distance"].transform("mean")
    df["dist_ratio_travel"] = df["Flight Distance"] / df.groupby("Type of Travel")["Flight Distance"].transform("mean")
    return df

BASE_NUM = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
            "service_sum", "service_mean", "service_std", "total_delay", "is_delayed"] + SERVICE_COLS
BASE_TE = CAT_COLS + ["travel_x_class", "travel_x_customer", "dist_bucket"]
NEW_NUM = ["digital_mean", "comfort_mean", "rest_mean", "digital_zero_cnt", "comfort_zero_cnt",
           "digital_mean_nz", "dist_ratio_class", "dist_ratio_travel"] + [f"{c[:12]}_is0" for c in SERVICE_COLS]

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

def run(name, train_df, test_df, num_cols, te_cols):
    lgb_params = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
                      min_child_samples=50, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                      reg_alpha=0.3, reg_lambda=1.0, n_estimators=8000, random_state=SEED,
                      n_jobs=-1, verbosity=-1)
    oof = np.zeros(len(train_df))
    for tr_idx, va_idx in skf.split(train_df, y):
        X_tr, X_va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        parts_tr = [X_tr[num_cols].reset_index(drop=True)]
        parts_va = [X_va[num_cols].reset_index(drop=True)]
        parts_te = [test_df[num_cols].reset_index(drop=True)]
        if te_cols:
            te_tr, te_va, te_te = te_encode(X_tr, X_va, test_df, y_tr, te_cols)
            parts_tr.append(te_tr.reset_index(drop=True))
            parts_va.append(te_va.reset_index(drop=True))
            parts_te.append(te_te.reset_index(drop=True))
        Xt = pd.concat(parts_tr, axis=1)
        Xv = pd.concat(parts_va, axis=1)
        Xte = pd.concat(parts_te, axis=1)
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(Xt, y_tr, eval_X=Xv, eval_y=y_va, callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va_idx] = m.predict_proba(Xv)[:, 1]
    auc = roc_auc_score(y, oof)
    print(f"  {name:40s} OOF = {auc:.5f}", flush=True)
    return auc

base_tr = build_base(train); base_te = build_base(test)
for c in BASE_NUM:
    if base_tr[c].isna().any(): base_tr[c] = base_tr[c].fillna(0)
    if base_te[c].isna().any(): base_te[c] = base_te[c].fillna(0)
new_tr = add_insight_feats(base_tr); new_te = add_insight_feats(base_te)
for c in NEW_NUM:
    if new_tr[c].isna().any(): new_tr[c] = new_tr[c].fillna(0)
    if new_te[c].isna().any(): new_te[c] = new_te[c].fillna(0)

print("=== 洞察特征实验 (LGBM 5折) ===", flush=True)
run("A. 基线", base_tr, base_te, BASE_NUM, BASE_TE)
run("B. +洞察特征", new_tr, new_te, BASE_NUM + NEW_NUM, BASE_TE)
