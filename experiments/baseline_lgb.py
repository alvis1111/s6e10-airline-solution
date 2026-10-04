"""S6E10 Airline Satisfaction — baseline: OOF target encoding + LightGBM.

Playbook 沿用 S6E9: 类别特征做 out-of-fold target encoding, 数值特征直接喂,
LightGBM + StratifiedKFold, 用 OOF AUC 作为可靠验证信号 (别只看 public LB).
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

# ---- 读数据 ----
train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
sub = pd.read_csv(f"{DATA}/sample_submission.csv")

y = train[TARGET].astype(int)

cat_cols = ["Gender", "Customer Type", "Type of Travel", "Class"]
num_cols = [c for c in train.columns if c not in [TARGET, "id"] + cat_cols]

# ---- 缺失处理 (Arrival Delay in Minutes 少量缺失) ----
for df in (train, test):
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(0)

# ---- OOF 目标编码 (smoothing 防过拟合, 每折只用该折训练部分算 mean) ----
def smoothed_te(tr, va, y_tr, y_va, prior, m=20.0):
    """tr/va: 该特征的原始值; 返回 (train编码, valid编码)."""
    stats = (
        pd.DataFrame({"v": tr, "y": y_tr})
        .groupby("v")["y"]
        .agg(["mean", "count"])
    )
    enc = (stats["mean"] * stats["count"] + prior * m) / (stats["count"] + m)
    return tr.map(enc), va.map(enc)

# 先把类别特征编码成 0..k-1, 供 target encoding 分组用
for c in cat_cols:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype("category")

# ---- 训练 ----
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(train))
pred = np.zeros(len(test))
test_feats_cache = None

lgb_params = dict(
    objective="binary",
    metric="auc",
    learning_rate=0.05,
    num_leaves=255,
    max_depth=-1,
    min_child_samples=40,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_estimators=5000,
    random_state=SEED,
    n_jobs=-1,
    verbosity=-1,
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
    X_tr = train.iloc[tr_idx].copy()
    X_va = train.iloc[va_idx].copy()
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    # 该折内做 target encoding
    prior = y_tr.mean()
    X_tr_enc, X_va_enc = X_tr[cat_cols].copy(), X_va[cat_cols].copy()
    for c in cat_cols:
        X_tr_enc[c], X_va_enc[c] = smoothed_te(
            X_tr[c], X_va[c], y_tr, y_va, prior
        )

    X_tr_full = pd.concat([X_tr[num_cols], X_tr_enc], axis=1)
    X_va_full = pd.concat([X_va[num_cols], X_va_enc], axis=1)

    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(
        X_tr_full, y_tr,
        eval_set=[(X_va_full, y_va)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    oof[va_idx] = model.predict_proba(X_va_full)[:, 1]

    # 测试集: 用本折训练部分算的 target encoding (prior = 本折均值)
    prior_full = y_tr.mean()
    X_te_enc = test[cat_cols].copy()
    for c in cat_cols:
        stats = (
            pd.DataFrame({"v": X_tr[c], "y": y_tr})
            .groupby("v")["y"]
            .agg(["mean", "count"])
        )
        te_map = (stats["mean"] * stats["count"] + prior_full * 20.0) / (
            stats["count"] + 20.0
        )
        X_te_enc[c] = test[c].map(te_map).fillna(prior_full)

    X_te_full = pd.concat([test[num_cols], X_te_enc], axis=1)
    pred += model.predict_proba(X_te_full)[:, 1] / N_FOLDS

    print(
        f"fold {fold}: best_iter={model.best_iteration_:>4}  "
        f"val_auc={roc_auc_score(y_va, oof[va_idx]):.5f}"
    )

auc = roc_auc_score(y, oof)
print(f"\nOOF AUC: {auc:.5f}")

# ---- 提交 ----
sub[TARGET] = pred
sub.to_csv("submission_lgb_baseline.csv", index=False)
print(f"submission saved -> submission_lgb_baseline.csv  ({len(sub)} rows)")
print(f"pred range: [{pred.min():.4f}, {pred.max():.4f}], mean={pred.mean():.4f}")
