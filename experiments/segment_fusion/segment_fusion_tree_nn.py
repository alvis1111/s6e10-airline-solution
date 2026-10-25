"""segment_fusion_tree_nn — 完整树/NN 两家族基线上的分组融合筛查。

固定 4 组: (Wi-Fi == 4) × (Online boarding == 4)。
家族聚合: F_tree = mean logit(14 树成员), F_nn = mean logit(10 RealMLP 成员)。
三融合器(嵌套 5 折 LR, C=1.0):
  A: 全局 LR on [F_tree, F_nn]
  B: A + 群体 one-hot
  C: B + 群体 × (F_tree - F_nn)
"""
import json, os, time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

DATA = r"C:\Users\GEM07\kaggle\s6e10-airline\data\playground-series-s6e10"
S7 = r"C:\Users\GEM07\.cache\kagglehub\datasets\sachith7\s6e10-stack-oof-predictions\versions\3"

train = pd.read_csv(f"{DATA}/train.csv")
y = train['satisfaction'].to_numpy(np.int64)
wifi = train['Inflight wifi service'].to_numpy()
boarding = train['Online boarding'].to_numpy()
g = (wifi == 4).astype(int) * 2 + (boarding == 4).astype(int)

logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
mj = json.load(open(os.path.join(S7, "members.json"), encoding="utf-8"))

tree_logits, nn_logits = [], []
for name, meta in mj.items():
    oof = np.load(os.path.join(S7, name + ".npz"))["oof"]
    if meta["model"] in ("xgb", "lgb", "cat"):
        tree_logits.append(logit(oof))
    elif meta["model"] == "realmlp":
        nn_logits.append(logit(oof))

F_tree = np.mean(tree_logits, axis=0)
F_nn = np.mean(nn_logits, axis=0)
fd = F_tree - F_nn
print(f"树家族 {len(tree_logits)} 成员, NN家族 {len(nn_logits)} 成员")
print(f"AUC(F_tree)={roc_auc_score(y, 1/(1+np.exp(-F_tree))):.5f}  AUC(F_nn)={roc_auc_score(y, 1/(1+np.exp(-F_nn))):.5f}  corr={np.corrcoef(F_tree, F_nn)[0,1]:.6f}")

G = np.zeros((len(y), 4)); G[np.arange(len(y)), g] = 1.0
FA = np.column_stack([F_tree, F_nn])
FB = np.column_stack([F_tree, F_nn, G[:, 1], G[:, 2], G[:, 3]])
FC = np.column_stack([F_tree, F_nn, G[:, 1], G[:, 2], G[:, 3],
                      G[:, 0] * fd, G[:, 1] * fd, G[:, 2] * fd, G[:, 3] * fd])

skf = StratifiedKFold(5, shuffle=True, random_state=42)
def cv_auc(F):
    oof = np.zeros(len(y))
    for tr, va in skf.split(F, y):
        m = LogisticRegression(C=1.0, max_iter=3000)
        m.fit(F[tr], y[tr])
        oof[va] = m.predict_proba(F[va])[:, 1]
    return roc_auc_score(y, oof), oof

auc_A, oof_A = cv_auc(FA)
auc_B, oof_B = cv_auc(FB)
auc_C, oof_C = cv_auc(FC)
print(f"\n嵌套5折 OOF AUC:  A={auc_A:.6f}  B={auc_B:.6f}  C={auc_C:.6f}")
print(f"  Δ(B-A) 群体偏差 = {auc_B-auc_A:+.6f}")
print(f"  Δ(C-A) 群体权重 = {auc_C-auc_A:+.6f}")
print(f"  Δ(C-B) 群体×家族差 = {auc_C-auc_B:+.6f}")

def auc_fast(yy, s):
    r = rankdata(s); n1 = yy.sum(); n0 = len(yy) - n1
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
rng = np.random.default_rng(42); B = 800; n = len(y)
deltas = np.empty(B)
for b in range(B):
    idx = rng.integers(0, n, n)
    deltas[b] = auc_fast(y[idx], oof_C[idx]) - auc_fast(y[idx], oof_A[idx])
print(f"\n配对 bootstrap (C vs A, B={B}): mean Δ={deltas.mean():+.6f}  SE={deltas.std(ddof=1):.6f}")
print(f"  95% CI = [{np.percentile(deltas, 2.5):+.6f}, {np.percentile(deltas, 97.5):+.6f}]  P(Δ>0)={(deltas > 0).mean():.3f}")
