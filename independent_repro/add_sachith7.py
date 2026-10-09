"""加入 sachith7 的 10 折成员到 53 成员 logistic 融合，检查边际增益。

步骤：核对重复成员 → 复现 53 成员 logistic 融合(5折seed33,C=1.0) → 加入 10 个 sachith7 10折成员
→ 同划分/正则/评估重拟合 → 逐折增益 + 配对 DeLong。
"""
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from delong import delong_ci

train = pd.read_csv("data/playground-series-s6e10/train.csv")
y = train["satisfaction"].to_numpy(np.int64)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

# 48 公开成员
MEM = r"C:/Users/GEM07/.cache/kagglehub/notebooks/goodpjw2008/s6e10-tabpfn-route-categories-lb-0-96160/output/versions/1"
mo = pd.read_csv(MEM + "/oof_members.csv").set_index("id").reindex(train["id"])
pub = np.column_stack([logit(mo[c].to_numpy()) for c in mo.columns])
# 5 自有成员
z = np.load("pulled_busy53/oof_ours.npz", allow_pickle=False)
own = np.column_stack([logit(z[n]) for n in ["lgbm", "lgbm_xt", "cat", "xgb", "realmlp"]])
X53 = np.column_stack([pub, own])
_names53 = list(mo.columns) + ["lgbm", "lgbm_xt", "cat", "xgb", "realmlp"]
print(f"53 成员: {X53.shape}", flush=True)

# 10 个 sachith7 10 折成员
SA = r"C:/Users/GEM07/.cache/kagglehub/datasets/sachith7/s6e10-stack-oof-predictions/versions/2"
sa_files = sorted(glob.glob(SA + "/*_k10*.npz"))
print(f"sachith7 10折成员数: {len(sa_files)}", flush=True)

# 重复成员核对 + 收集
sa_cols, dup_flags = [], []
for f in sa_files:
    nm = "sa_" + f.split("/")[-1][:-4]
    oof = np.load(f, allow_pickle=False)["oof"]
    assert len(oof) == len(train), f"{nm} 长度 {len(oof)}"
    lg = logit(oof)
    corrs = [np.corrcoef(lg, X53[:, j])[0, 1] for j in range(X53.shape[1])]
    max_corr = max(corrs); max_name = _names53[int(np.argmax(corrs))]
    dup_flags.append((nm, max_corr, max_name))
    sa_cols.append(lg)
    print(f"  {nm:28s} OOF AUC {roc_auc_score(y, oof):.6f}  最近 {max_name} 相关 {max_corr:.4f}", flush=True)

Xsa = np.column_stack(sa_cols)
X63 = np.column_stack([X53, Xsa])
print(f"\n63 成员: {X63.shape}", flush=True)

FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=33).split(X63, y))


def nested_logreg(X, folds):
    oof = np.zeros(len(y))
    for ia, iv in folds:
        m = LogisticRegression(max_iter=2000, C=1.0).fit(X[ia], y[ia])
        oof[iv] = m.decision_function(X[iv])
    return oof


oof53 = nested_logreg(X53, FOLDS)
oof63 = nested_logreg(X63, FOLDS)
print(f"\n53 成员 logistic OOF = {roc_auc_score(y, oof53):.6f}", flush=True)
print(f"63 成员 logistic OOF = {roc_auc_score(y, oof63):.6f}", flush=True)
_, _, delta, lo, hi, p = delong_ci(y, oof53, oof63)
print(f"配对 DeLong: Δ={delta:+.7f}  95%CI=[{lo:+.7f}, {hi:+.7f}]  p={p:.4f}", flush=True)
