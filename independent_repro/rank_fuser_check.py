"""单折正则尺度 + 收敛对照：pairwise 排序损失在不同 l2 下 vs logistic 回归。

修正 rank_fuser.py 的混杂：之前 l2=0.01，而 logistic 回归的有效 L2≈1/(2·C·n)≈8.9e-7，
差 ~1.1 万倍。这里只在 fold 0 上扫 l2，并报告 L-BFGS 收敛信息。
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("data/playground-series-s6e10/train.csv")
y = train["satisfaction"].to_numpy(np.int64)

MEM = r"C:/Users/GEM07/.cache/kagglehub/notebooks/goodpjw2008/s6e10-tabpfn-route-categories-lb-0-96160/output/versions/1"
mo = pd.read_csv(MEM + "/oof_members.csv").set_index("id").reindex(train["id"])
z = np.load("pulled_busy53/oof_ours.npz", allow_pickle=False)
own_names = ["lgbm", "lgbm_xt", "cat", "xgb", "realmlp"]
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
X = np.column_stack([logit(mo[c].to_numpy()) for c in mo.columns] + [logit(z[n]) for n in own_names])

FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=33).split(X, y))
ia, iv = FOLDS[0]
Xtr, ytr, Xva, yva = X[ia], y[ia], X[iv], y[iv]

m = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
lr_auc = roc_auc_score(yva, m.decision_function(Xva))
print(f"logistic (C=1.0) fold0 AUC = {lr_auc:.6f}", flush=True)

rng = np.random.RandomState(0)
pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
n_pairs = 2_000_000
pi = rng.choice(pos, n_pairs, replace=True)
nj = rng.choice(neg, n_pairs, replace=True)
Xp = Xtr[pi]; Xn = Xtr[nj]


def fit(l2, maxiter=400):
    def loss_grad(w):
        d = (Xp - Xn) @ w
        sig = 1.0 / (1.0 + np.exp(d))
        loss = float(np.log(1 + np.exp(-d)).mean() + l2 * (w @ w))
        grad = -(sig[:, None] * (Xp - Xn)).mean(axis=0) + 2 * l2 * w
        return loss, grad
    res = minimize(loss_grad, np.zeros(Xtr.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": 1e-12})
    gnorm = float(np.linalg.norm(res.jac)) if res.jac is not None else float("nan")
    return Xva @ res.x, res


print(f"\n{'l2':>10s} {'AUC':>9s} {'Δlr':>8s} {'success':>8s} {'nit':>5s} {'|g|':>10s}", flush=True)
for l2 in [0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7, 1e-8, 0.0]:
    pred, res = fit(l2)
    auc = roc_auc_score(yva, pred)
    print(f"{l2:>10.0e} {auc:>9.6f} {auc-lr_auc:>+8.6f} {str(res.success):>8s} {res.nit:>5d} "
          f"{np.linalg.norm(res.jac) if res.jac is not None else float('nan'):>10.2e}", flush=True)
