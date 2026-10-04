"""S6E10 — 特征穷举实验: 评分聚合 / 数值变换 / 交互目标编码, 逐组测 OOF."""
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

def build_base(df):
    df = df.copy()
    df["service_sum"] = df[SERVICE_COLS].sum(axis=1)
    df["service_mean"] = df[SERVICE_COLS].mean(axis=1)
    df["service_std"] = df[SERVICE_COLS].std(axis=1)
    df["total_delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    df["is_delayed"] = (df["total_delay"] > 0).astype(int)
    df["travel_x_class"] = df["Type of Travel"] + "_" + df["Class"]
    df["travel_x_customer"] = df["Type of Travel"] + "_" + df["Customer Type"]
    df["dist_bucket"] = pd.cut(
        df["Flight Distance"], bins=[0, 500, 1000, 2000, 3000, 5000, np.inf], labels=False)
    return df

def add_num_feats(df):
    df = df.copy()
    S = SERVICE_COLS
    df["service_min"] = df[S].min(axis=1)
    df["service_max"] = df[S].max(axis=1)
    df["service_median"] = df[S].median(axis=1)
    df["service_range"] = df["service_max"] - df["service_min"]
    df["count_zero"] = (df[S] == 0).sum(axis=1)
    df["count_low"] = (df[S] <= 2).sum(axis=1)
    df["count_high"] = (df[S] >= 4).sum(axis=1)
    df["log_distance"] = np.log1p(df["Flight Distance"])
    df["delay_ratio"] = df["total_delay"] / (df["Flight Distance"] + 1)
    df["online_share"] = df["Online boarding"] / (df["service_sum"] + 1e-6)
    return df

def add_interaction_feats(df):
    df = df.copy()
    df["ob_bucket"] = pd.cut(df["Online boarding"], bins=[-1, 1, 3, 5], labels=["low", "mid", "high"])
    df["svc_bucket"] = pd.cut(df["service_mean"], bins=[-0.1, 2.5, 3.5, 5.1], labels=["low", "mid", "high"])
    df["class_x_ob"] = df["Class"].astype(str) + "_" + df["ob_bucket"].astype(str)
    df["travel_x_ob"] = df["Type of Travel"].astype(str) + "_" + df["ob_bucket"].astype(str)
    df["class_x_dist"] = df["Class"].astype(str) + "_" + df["dist_bucket"].astype(str)
    df["class_x_svc"] = df["Class"].astype(str) + "_" + df["svc_bucket"].astype(str)
    df["class_x_delay"] = df["Class"].astype(str) + "_" + df["is_delayed"].astype(str)
    df["travel_class_ob"] = df["Type of Travel"].astype(str) + "_" + df["Class"].astype(str) + "_" + df["ob_bucket"].astype(str)
    return df

BASE_NUM = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
            "service_sum", "service_mean", "service_std", "total_delay", "is_delayed"] + SERVICE_COLS
BASE_TE = CAT_COLS + ["travel_x_class", "travel_x_customer", "dist_bucket"]

NEW_NUM = ["service_min", "service_max", "service_median", "service_range",
           "count_zero", "count_low", "count_high", "log_distance", "delay_ratio", "online_share"]
NEW_TE = ["class_x_ob", "travel_x_ob", "class_x_dist", "class_x_svc", "class_x_delay", "travel_class_ob"]

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
    print(f"  {name:45s} OOF = {auc:.5f}", flush=True)
    return auc

# 构造三套特征
base_tr = build_base(train); base_te = build_base(test)
for c in BASE_NUM:
    if base_tr[c].isna().any(): base_tr[c] = base_tr[c].fillna(0)
    if base_te[c].isna().any(): base_te[c] = base_te[c].fillna(0)

num_tr = add_num_feats(base_tr); num_te = add_num_feats(base_te)
inter_tr = add_interaction_feats(num_tr); inter_te = add_interaction_feats(num_te)

print("=== 特征穷举实验 (LGBM 5折) ===", flush=True)
run("A. 基线", base_tr, base_te, BASE_NUM, BASE_TE)
run("B. +评分聚合/数值变换", num_tr, num_te, BASE_NUM + NEW_NUM, BASE_TE)
run("C. +交互目标编码", inter_tr, inter_te, BASE_NUM + NEW_NUM, BASE_TE + NEW_TE)
