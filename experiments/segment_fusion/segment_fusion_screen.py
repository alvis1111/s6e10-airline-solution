"""segment_fusion_screen — 按乘客群体调融合权重的筛查。

固定 4 组: (WiFi 评分 == 4) × (Online boarding == 4)。不搜索其他分组。
三个融合器(其余规则一致, 嵌套 5 折 LR, C=1.0):
  A: 全局 LR(现有全局权重)
  B: A + 群体 one-hot(群体概率偏差)
  C: B + 群体标记 × 模型家族预测差(群体是否需要不同权重)
比较全部验证样本的 AUC(含跨群体排序)。
"""
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

DATA = r"C:\Users\GEM07\kaggle\s6e10-airline\data\playground-series-s6e10"
train = pd.read_csv(f"{DATA}/train.csv")
y = train['satisfaction'].to_numpy(np.int64)
wifi = train['Inflight wifi service'].to_numpy()
boarding = train['Online boarding'].to_numpy()

g = (wifi == 4).astype(int) * 2 + (boarding == 4).astype(int)   # 0..3
print("群体分布 (wifi==4, board==4):")
for k in range(4):
    print(f"  g{k} (wifi==4={k//2}, board==4={k%2}): {int((g==k).sum())} 行, 正例率={(y[g==k].mean() if (g==k).sum() else 0):.4f}")

route_l = np.load("cache/oof_route_l.npy")
route_x = np.load("cache/oof_route_x.npy")
L = np.load("cache/oof_L_te21.npy")
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
l, x, ll = logit(route_l), logit(route_x), logit(L)
route_blend = 0.5 * l + 0.5 * x
fd = route_blend - ll          # 家族预测差 (树 route 家族 vs 线性叶 L)

G = np.zeros((len(y), 4)); G[np.arange(len(y)), g] = 1.0
FA = np.column_stack([l, x, ll])
FB = np.column_stack([l, x, ll, G[:, 1], G[:, 2], G[:, 3]])                       # 群体偏差 (g0 为基线)
FC = np.column_stack([l, x, ll, G[:, 1], G[:, 2], G[:, 3],
                      G[:, 0] * fd, G[:, 1] * fd, G[:, 2] * fd, G[:, 3] * fd])    # + 群体×家族差

skf = StratifiedKFold(5, shuffle=True, random_state=42)
def cv_auc(F):
    oof = np.zeros(len(y))
    per_fold = []
    for tr, va in skf.split(F, y):
        m = LogisticRegression(C=1.0, max_iter=3000)
        m.fit(F[tr], y[tr])
        oof[va] = m.predict_proba(F[va])[:, 1]
        per_fold.append((va, roc_auc_score(y[va], oof[va])))
    return roc_auc_score(y, oof), oof, per_fold

auc_A, oof_A, pfA = cv_auc(FA)
auc_B, oof_B, pfB = cv_auc(FB)
auc_C, oof_C, pfC = cv_auc(FC)
print(f"\n嵌套5折 OOF AUC:  A={auc_A:.6f}  B={auc_B:.6f}  C={auc_C:.6f}")
print(f"  Δ(B-A) 群体偏差 = {auc_B-auc_A:+.6f}")
print(f"  Δ(C-A) 群体权重 = {auc_C-auc_A:+.6f}")
print(f"  Δ(C-B) 群体×家族差 = {auc_C-auc_B:+.6f}")

# 配对 bootstrap (A vs C)
def auc_fast(yy, s):
    r = rankdata(s); n1 = yy.sum(); n0 = len(yy)-n1
    return (r[yy==1].sum() - n1*(n1+1)/2.0) / (n1*n0)
rng = np.random.default_rng(42); B = 800; n = len(y)
deltas = np.empty(B)
for b in range(B):
    idx = rng.integers(0, n, n)
    deltas[b] = auc_fast(y[idx], oof_C[idx]) - auc_fast(y[idx], oof_A[idx])
print(f"\n配对 bootstrap (C vs A, B={B}): mean Δ={deltas.mean():+.6f}  SE={deltas.std(ddof=1):.6f}")
print(f"  95% CI = [{np.percentile(deltas,2.5):+.6f}, {np.percentile(deltas,97.5):+.6f}]  P(Δ>0)={(deltas>0).mean():.3f}")
