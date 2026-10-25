"""segment_fusion_3fam — 完整三家族(树/NN/TabPFN)基线上的分组融合筛查。

固定 4 组: (Wi-Fi == 4) × (Online boarding == 4)。
家族聚合: F_tree(14 树), F_nn(10 RealMLP), F_tabpfn(8 TabPFN)。
三融合器(嵌套 5 折 LR, C=1.0):
  A: 全局 LR on [F_tree, F_nn, F_tabpfn]
  B: A + 群体 one-hot
  C: B + 群体 × (F_nn-F_tree) + 群体 × (F_tabpfn-F_tree)
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

DATA = r"C:\Users\GEM07\kaggle\s6e10-airline\data\playground-series-s6e10"
S7 = r"C:\Users\GEM07\.cache\kagglehub\datasets\sachith7\s6e10-stack-oof-predictions\versions\3"
GP = r"C:\Users\GEM07\.cache\kagglehub\datasets\goodpjw2008\s6e10-tabpfn-member\versions\2"
CT = r"C:\Users\GEM07\.cache\kagglehub\datasets\cdeotte\s6e10-tfm-oof-and-test-predictions\versions\1"

train = pd.read_csv(f"{DATA}/train.csv")
y = train['satisfaction'].to_numpy(np.int64)
wifi = train['Inflight wifi service'].to_numpy()
boarding = train['Online boarding'].to_numpy()
g = (wifi == 4).astype(int) * 2 + (boarding == 4).astype(int)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

# 树 + NN (Sachith7)
mj = json.load(open(os.path.join(S7, "members.json"), encoding="utf-8"))
tree_logits, nn_logits = [], []
for name, meta in mj.items():
    oof = np.load(os.path.join(S7, name + ".npz"))["oof"]
    if meta["model"] in ("xgb", "lgb", "cat"):
        tree_logits.append(logit(oof))
    elif meta["model"] == "realmlp":
        nn_logits.append(logit(oof))

# TabPFN (goodpjw2008 csv + cdeotte tabpfn35)
tabpfn_csvs = ["oof_tabpfn_te4.csv", "oof_tabpfn_te.csv", "oof_tabpfn_catfd10.csv", "oof_tabpfn_catfd.csv",
               "oof_tabpfn_full_raw.csv", "oof_tabpfn_full_raw_s1.csv", "oof_tabpfn_catfd_s1.csv"]
tabpfn_logits = []
for f in tabpfn_csvs:
    d = pd.read_csv(os.path.join(GP, f)).set_index("id").reindex(train["id"])["pred"].to_numpy()
    tabpfn_logits.append(logit(d))
tabpfn_logits.append(logit(np.load(os.path.join(CT, "oof_tabpfn35.npy"))))

F_tree = np.mean(tree_logits, axis=0)
F_nn = np.mean(nn_logits, axis=0)
F_tabpfn = np.mean(tabpfn_logits, axis=0)
print(f"树 {len(tree_logits)}, NN {len(nn_logits)}, TabPFN {len(tabpfn_logits)} 成员")
print(f"AUC: F_tree={roc_auc_score(y,1/(1+np.exp(-F_tree))):.5f}  F_nn={roc_auc_score(y,1/(1+np.exp(-F_nn))):.5f}  F_tabpfn={roc_auc_score(y,1/(1+np.exp(-F_tabpfn))):.5f}")
print(f"corr: tree-nn={np.corrcoef(F_tree,F_nn)[0,1]:.6f}  tree-tabpfn={np.corrcoef(F_tree,F_tabpfn)[0,1]:.6f}  nn-tabpfn={np.corrcoef(F_nn,F_tabpfn)[0,1]:.6f}")

fd_nn = F_nn - F_tree
fd_tab = F_tabpfn - F_tree
G = np.zeros((len(y), 4)); G[np.arange(len(y)), g] = 1.0

FA = np.column_stack([F_tree, F_nn, F_tabpfn])
FB = np.column_stack([F_tree, F_nn, F_tabpfn, G[:, 1], G[:, 2], G[:, 3]])
FC = np.column_stack([F_tree, F_nn, F_tabpfn, G[:, 1], G[:, 2], G[:, 3],
                      G[:, 0]*fd_nn, G[:, 1]*fd_nn, G[:, 2]*fd_nn, G[:, 3]*fd_nn,
                      G[:, 0]*fd_tab, G[:, 1]*fd_tab, G[:, 2]*fd_tab, G[:, 3]*fd_tab])

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
print(f"  Δ(B-A) = {auc_B-auc_A:+.6f}")
print(f"  Δ(C-A) = {auc_C-auc_A:+.6f}")
print(f"  Δ(C-B) = {auc_C-auc_B:+.6f}")

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
