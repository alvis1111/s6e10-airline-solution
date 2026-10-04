"""S6E10 — 分组模型测试: Personal Travel 专用模型 vs 全局模型."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

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
    return df

train = build(train); test = build(test)
NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes",
            "service_sum", "service_mean", "service_std", "total_delay", "is_delayed"] + SERVICE_COLS
TE_COLS = CAT_COLS + ["travel_x_class", "travel_x_customer", "dist_bucket"]
for df in (train, test):
    for c in NUM_COLS:
        if df[c].isna().any(): df[c] = df[c].fillna(0)
    for c in TE_COLS: df[c] = df[c].astype("category")

def te_encode(X_tr, X_va, X_te, y_tr, cols):
    prior = y_tr.mean()
    out_tr = X_tr[cols].astype(str).copy(); out_va = X_va[cols].astype(str).copy(); out_te = X_te[cols].astype(str).copy()
    for c in cols:
        stats = pd.DataFrame({"v": X_tr[c].astype(str), "y": y_tr}).groupby("v")["y"].agg(["mean","count"])
        enc = (stats["mean"]*stats["count"] + prior*20.0)/(stats["count"]+20.0)
        out_tr[c] = X_tr[c].astype(str).map(enc).fillna(prior)
        out_va[c] = X_va[c].astype(str).map(enc).fillna(prior)
        out_te[c] = X_te[c].astype(str).map(enc).fillna(prior)
    return out_tr.astype("float32"), out_va.astype("float32"), out_te.astype("float32")

lgb_params = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
                  min_child_samples=50, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                  reg_alpha=0.3, reg_lambda=1.0, n_estimators=8000, random_state=SEED, n_jobs=-1, verbosity=-1)

def train_oof(sub_df, sub_y, use_te=True):
    """在 sub_df 上 5 折, 返回 OOF."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(sub_df))
    for tr_idx, va_idx in skf.split(sub_df, sub_y):
        X_tr, X_va = sub_df.iloc[tr_idx], sub_df.iloc[va_idx]
        y_tr, y_va = sub_y[tr_idx], sub_y[va_idx]
        if use_te:
            te_tr, te_va, _ = te_encode(X_tr, X_va, test, y_tr, TE_COLS)
            Xt = pd.concat([X_tr[NUM_COLS].reset_index(drop=True), te_tr.reset_index(drop=True)], axis=1)
            Xv = pd.concat([X_va[NUM_COLS].reset_index(drop=True), te_va.reset_index(drop=True)], axis=1)
        else:
            Xt = X_tr[NUM_COLS].reset_index(drop=True); Xv = X_va[NUM_COLS].reset_index(drop=True)
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(Xt, y_tr, eval_X=Xv, eval_y=y_va, callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va_idx] = m.predict_proba(Xv)[:, 1]
    return oof

# 全局模型 (全量)
print("=== 全局模型 ===", flush=True)
oof_global = train_oof(train, y)
print(f"整体 AUC = {roc_auc_score(y, oof_global):.5f}", flush=True)
pers_mask = train["Type of Travel"] == "Personal Travel"
print(f"Personal 段 AUC (全局模型) = {roc_auc_score(y[pers_mask], oof_global[pers_mask]):.5f}", flush=True)

# Personal-only 模型
print("\n=== Personal-only 模型 ===", flush=True)
pers_df = train[pers_mask].reset_index(drop=True)
pers_y = y[pers_mask]
oof_pers = train_oof(pers_df, pers_y)
print(f"Personal-only AUC = {roc_auc_score(pers_y, oof_pers):.5f}", flush=True)

# 混合: Personal 用 personal模型, 其他用全局模型, 段内 rank 归一后合并
print("\n=== 混合排名 (分组模型 vs 全局) ===", flush=True)
pers_rank = rankdata(oof_pers) / len(pers_y)
biz_rank = rankdata(oof_global[~pers_mask]) / len(oof_global[~pers_mask])
comb = np.zeros(len(y))
comb[pers_mask] = pers_rank
comb[~pers_mask] = biz_rank
print(f"混合后整体 AUC (段内 rank 归一) = {roc_auc_score(y, comb):.5f}", flush=True)

# 全局 rank 归一 (对照: 同样段内归一但不换模型)
glob_rank = np.zeros(len(y))
glob_rank[pers_mask] = rankdata(oof_global[pers_mask]) / len(oof_global[pers_mask])
glob_rank[~pers_mask] = rankdata(oof_global[~pers_mask]) / len(oof_global[~pers_mask])
print(f"全局模型段内 rank 归一 AUC = {roc_auc_score(y, glob_rank):.5f}", flush=True)
