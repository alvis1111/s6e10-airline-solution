"""S6E10 — improved: 特征工程 + 交互目标编码 + LightGBM.

新增:
1. 服务评分聚合 (service_sum / service_mean / service_std)
2. 延误 -> 是否延误标志 + 总延误
3. Type of Travel x Class 交互 -> 单独 target encode
4. Flight Distance 分桶
"""
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
    # 服务评分聚合
    df["service_sum"] = df[SERVICE_COLS].sum(axis=1)
    df["service_mean"] = df[SERVICE_COLS].mean(axis=1)
    df["service_std"] = df[SERVICE_COLS].std(axis=1)
    # 延误
    df["total_delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    df["is_delayed"] = (df["total_delay"] > 0).astype(int)
    # 交互
    df["travel_x_class"] = df["Type of Travel"] + "_" + df["Class"]
    df["travel_x_customer"] = df["Type of Travel"] + "_" + df["Customer Type"]
    # 距离分桶 (商务长途占比高)
    df["dist_bucket"] = pd.cut(
        df["Flight Distance"], bins=[0, 500, 1000, 2000, 3000, 5000, np.inf], labels=False
    ).astype("category")
    # 各评分列是 ordinal, 保留原值
    return df

train = build_features(train)
test = build_features(test)

NUM_COLS = [
    "Age", "Flight Distance", "Departure Delay in Minutes",
    "Arrival Delay in Minutes", "service_sum", "service_mean",
    "service_std", "total_delay", "is_delayed",
] + SERVICE_COLS

# 需要 target encoding 的列 (含交互列 + 距离桶)
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

lgb_params = dict(
    objective="binary", metric="auc",
    learning_rate=0.03, num_leaves=127, max_depth=-1,
    min_child_samples=50, colsample_bytree=0.7, subsample=0.8,
    subsample_freq=1, reg_alpha=0.3, reg_lambda=1.0,
    n_estimators=8000, random_state=SEED, n_jobs=-1, verbosity=-1,
)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
pred = np.zeros(len(test))

for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
    X_tr, X_va = train.iloc[tr_idx], train.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    prior = y_tr.mean()
    tr_te = X_tr[TE_COLS].copy()
    va_te = X_va[TE_COLS].copy()
    te_te = test[TE_COLS].copy()
    for c in TE_COLS:
        tr_te[c], va_te[c] = smoothed_te(X_tr[c], X_va[c], y_tr, prior)
        stats = pd.DataFrame({"v": X_tr[c], "y": y_tr}).groupby("v")["y"].agg(["mean", "count"])
        te_map = (stats["mean"] * stats["count"] + prior * 20.0) / (stats["count"] + 20.0)
        te_te[c] = test[c].map(te_map).fillna(prior)

    X_tr_full = pd.concat([X_tr[NUM_COLS].reset_index(drop=True), tr_te.reset_index(drop=True)], axis=1)
    X_va_full = pd.concat([X_va[NUM_COLS].reset_index(drop=True), va_te.reset_index(drop=True)], axis=1)
    X_te_full = pd.concat([test[NUM_COLS].reset_index(drop=True), te_te.reset_index(drop=True)], axis=1)

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr_full, y_tr,
        eval_X=X_va_full, eval_y=y_va,
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )
    oof[va_idx] = model.predict_proba(X_va_full)[:, 1]
    pred += model.predict_proba(X_te_full)[:, 1] / N_FOLDS
    print(f"fold {fold}: best_iter={model.best_iteration_:>4}  val_auc={roc_auc_score(y_va, oof[va_idx]):.5f}")

auc = roc_auc_score(y, oof)
print(f"\nOOF AUC: {auc:.5f}")

sub[TARGET] = pred
sub.to_csv("submission_lgb_improved.csv", index=False)
print("submission saved -> submission_lgb_improved.csv")

# 特征重要性
imp = pd.Series(model.feature_importances_, index=X_tr_full.columns).sort_values(ascending=False)
print("\nTop features (last fold):")
print(imp.head(20).round(0).to_string())
