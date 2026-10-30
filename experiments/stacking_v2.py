"""S6E10 — 二层融合(Stacking) 重验证：对齐划分 + 受约束凸组合 + 完整嵌套 CV。

修正之前 stacking.py 的两个漏洞：
  1. base 模型 OOF 必须来自【同一折划分】。这里用 5 折 seed 42 的对齐缓存：
       route LGBM(oof_knn_without) + context LGBM/XGB/CB(oof_cf_l/x/c)。
  2. stacker 用【受约束的凸组合】(w_i>=0, sum=1)，不是无约束 LR，防过拟合。
  3. 完整嵌套 CV：外层 5 折，每折在内层(train)用 log-loss 优化凸权重，外层(val)打分。

对比：最强单腿 / 50-50 prob 平均 / 50-50 rank 平均 / 受约束凸组合(嵌套CV)。
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from scipy.stats import rankdata
from scipy.optimize import minimize

train = pd.read_csv("data/playground-series-s6e10/train.csv")
y = train["satisfaction"].to_numpy(dtype=np.int64)

# 对齐的 base OOF（全部 5 折 seed 42）
MODELS = {
    "route_lgb":  np.load("cache/oof_knn_without.npy"),
    "context_lgb": np.load("cache/oof_cf_l.npy"),
    "context_xgb": np.load("cache/oof_cf_x.npy"),
    "context_cb":  np.load("cache/oof_cf_c.npy"),
}
names = list(MODELS.keys())
X = np.column_stack([MODELS[k] for k in names])   # (n, 4) 对齐的 OOF
n = len(y)

def r(p): return rankdata(p) / len(p)

print("=" * 62)
print("  对齐 base 单腿 OOF：")
for k in names:
    print(f"    {k:12s} {roc_auc_score(y, MODELS[k]):.5f}")

# ---- 受约束凸组合的 log-loss ----
def fit_convex_weights(Xtr, ytr):
    def neg_ll(w):
        p = np.clip(Xtr @ w, 1e-6, 1 - 1e-6)
        return -np.mean(ytr * np.log(p) + (1 - ytr) * np.log(1 - p))
    res = minimize(neg_ll, x0=np.full(Xtr.shape[1], 1 / Xtr.shape[1]),
                   method="SLSQP",
                   bounds=[(0, 1)] * Xtr.shape[1],
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return res.x

# ---- 完整嵌套 CV：外层 5 折，内层拟合凸权重，外层打分 ----
outer = StratifiedKFold(5, shuffle=True, random_state=42)
oof_stack = np.zeros(n)
weights_by_fold = []
for tr, va in outer.split(X, y):
    w = fit_convex_weights(X[tr], y[tr])
    weights_by_fold.append(w)
    oof_stack[va] = X[va] @ w

# ---- 参考基线 ----
best_single = max(roc_auc_score(y, MODELS[k]) for k in names)
prob_5050 = (X[:, 0] + X[:, [1, 2, 3]].mean(axis=1)) / 2   # route vs context 均值 50/50
rank_5050 = (r(X[:, 0]) + r(X[:, [1, 2, 3]].mean(axis=1))) / 2

print("-" * 62)
print(f"  最强单腿              = {best_single:.5f}")
print(f"  route vs context 50/50 prob = {roc_auc_score(y, prob_5050):.5f}")
print(f"  route vs context 50/50 rank = {roc_auc_score(y, rank_5050):.5f}")
print(f"  受约束凸组合(完整嵌套CV)   = {roc_auc_score(y, oof_stack):.5f}")
print("-" * 62)
print("  各折学到的凸权重 (route_lgb/context_lgb/context_xgb/context_cb)：")
for i, w in enumerate(weights_by_fold):
    print(f"    fold {i}: " + " ".join(f"{x:.3f}" for x in w))
print("  平均权重:", " ".join(f"{x:.3f}" for x in np.mean(weights_by_fold, axis=0)))

# 全量拟合（用于展示最终权重）
w_full = fit_convex_weights(X, y)
print(f"  全量拟合权重:", " ".join(f"{x:.3f}" for x in w_full))
print(f"  stack 相对最强单腿 delta = {roc_auc_score(y, oof_stack) - best_single:+.5f}")
