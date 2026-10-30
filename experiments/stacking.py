"""S6E10 — Stacking：用各模型的 OOF 预测训 meta-learner，替换手调 30/30/40 权重。

我们最终 LB 0.96100 是「航线画像 GBDT 30% + context_boost 30% + TabM 40%」的手调 rank-blend。
本脚本把这些腿的**个体模型** OOF 都拿来（14 个模型，含 GBDT/TabM 多家族），训一个
stacker（线性 LR / 非线性 LGBM），用嵌套 CV 得到诚实 OOF，和手调权重对比。

关键：stacker 的 OOF 必须用**嵌套 CV**（外层 fold 只训内层、预测外层），否则直接
在全量 OOF 上训练再打分会把 stacker 过拟合到 OOF，虚高。
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

DATA = "data/playground-series-s6e10"
train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
sub = pd.read_csv(f"{DATA}/sample_submission.csv")
y = train["satisfaction"].to_numpy(dtype=np.int64)

# 候选模型（有 oof + pred 的个体模型）
MODELS = {
    "route_x":        ("oof_route_x.npy",        "pred_route_x.npy"),
    "route_l":        ("oof_route_l.npy",        "pred_route_l.npy"),
    "route_tabm":     ("oof_route_tabm.npy",     "pred_route_tabm.npy"),
    "cf_x":           ("oof_cf_x.npy",           "pred_cf_x.npy"),
    "cf_l":           ("oof_cf_l.npy",           "pred_cf_l.npy"),
    "cf_c":           ("oof_cf_c.npy",           "pred_cf_c.npy"),
    "ms_l":           ("oof_ms_l.npy",           "pred_ms_l.npy"),
    "ms_x":           ("oof_ms_x.npy",           "pred_ms_x.npy"),
    "ms_c":           ("oof_ms_c.npy",           "pred_ms_c.npy"),
    "te_proper_lgb":  ("oof_te_proper_lgb.npy",  "pred_te_proper_lgb.npy"),
    "lgb_pte":        ("oof_lgb_pte.npy",        "pred_lgb_pte.npy"),
    "xgb_pte":        ("oof_xgb_pte.npy",        "pred_xgb_pte.npy"),
    "cb_pte":         ("oof_cb_pte.npy",         "pred_cb_pte.npy"),
}

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

# 加载 → logits
X = np.zeros((len(train), len(MODELS)), dtype=np.float64)
Xt = np.zeros((len(test), len(MODELS)), dtype=np.float64)
names = list(MODELS.keys())
for j, k in enumerate(names):
    X[:, j] = logit(np.load(f"cache/{MODELS[k][0]}"))
    Xt[:, j] = logit(np.load(f"cache/{MODELS[k][1]}"))

# ---- 参考基线 ----
def auc(oof):
    return roc_auc_score(y, oof)

# (a) 单模型最强（已知）
best_single = max(auc(X[:, j]) for j in range(len(names)))
# (b) 等权 logit 平均
eq = X.mean(axis=1)
# (c) 重构最终 30/30/40 rank-blend（腿 = route_leg / context_leg / tabm_leg）
route_leg = 0.5 * X[:, 1] + 0.5 * X[:, 0]              # route_l + route_x (logit)
context_leg = X[:, [3, 4, 5]].mean(axis=1)             # cf_x/cf_l/cf_c (logit)
tabm_leg = X[:, 2]                                     # route_tabm
r = lambda a: rankdata(a) / len(a)
final_340 = 0.3 * r(route_leg) + 0.3 * r(context_leg) + 0.4 * r(tabm_leg)

print("=" * 62)
print("  参考基线 (OOF AUC)")
print(f"  最强单模型              = {best_single:.5f}")
print(f"  等权 logit 平均 (14 模型) = {auc(eq):.5f}")
print(f"  重构 30/30/40 rank-blend  = {auc(final_340):.5f}")

# ---- 嵌套 CV 诚实评估 ----
outer = StratifiedKFold(5, shuffle=True, random_state=42)
oof_lr = np.zeros(len(train))
oof_lgb = np.zeros(len(train))

lr = LogisticRegression(max_iter=1000, C=1.0)   # 线性 stacker，L2 正则
for tr, va in outer.split(X, y):
    lr.fit(X[tr], y[tr])
    oof_lr[va] = lr.predict_proba(X[va])[:, 1]

lgb_stack = dict(objective="binary", metric="auc", n_estimators=400, learning_rate=0.03,
                 num_leaves=15, min_child_samples=2000, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbosity=-1)
for tr, va in outer.split(X, y):
    m = lgb.LGBMClassifier(**lgb_stack)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[va] = m.predict_proba(X[va])[:, 1]

print("-" * 62)
print("  Stacking (嵌套 CV, 诚实 OOF)")
print(f"  LR stacker  (线性)   = {auc(oof_lr):.5f}")
print(f"  LGBM stacker(非线性) = {auc(oof_lgb):.5f}")

# ---- 全量拟合 → 测试预测 + 提交 ----
lr_full = LogisticRegression(max_iter=1000, C=1.0).fit(X, y)
pred_lr = lr_full.predict_proba(Xt)[:, 1]

lgb_full = lgb.LGBMClassifier(**lgb_stack)
lgb_full.fit(X, y)
pred_lgb = lgb_full.predict_proba(Xt)[:, 1]

sub["satisfaction"] = pred_lr
sub.to_csv("submission_stacked_lr.csv", index=False)
sub["satisfaction"] = pred_lgb
sub.to_csv("submission_stacked_lgb.csv", index=False)

print("-" * 62)
print("  LR stacker 权重 (logit 系数):")
for k, w in zip(names, lr_full.coef_[0]):
    print(f"    {k:15s} {w:+.4f}")
print(f"  已保存 submission_stacked_lr.csv / submission_stacked_lgb.csv")
