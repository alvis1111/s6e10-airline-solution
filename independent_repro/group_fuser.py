"""分组正则化融合：对同配置不同种子的成员权重差异加惩罚，对比原 logistic 融合。

目标 = 0.5*||w||² + C*Σlog-loss + λ*w^T L w，L 是「同组成员全连接」图拉普拉斯。
保留负权重、不统一加强 L2。先两折，正增益再考虑五折。
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from delong import delong_ci

train = pd.read_csv("data/playground-series-s6e10/train.csv")
y = train["satisfaction"].to_numpy(np.int64)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

MEM = r"C:/Users/GEM07/.cache/kagglehub/notebooks/goodpjw2008/s6e10-tabpfn-route-categories-lb-0-96160/output/versions/1"
mo = pd.read_csv(MEM + "/oof_members.csv").set_index("id").reindex(train["id"])
z = np.load("pulled_busy53/oof_ours.npz", allow_pickle=False)
own_names = ["lgbm", "lgbm_xt", "cat", "xgb", "realmlp"]
names = list(mo.columns) + own_names
X = np.column_stack([logit(mo[c].to_numpy()) for c in mo.columns] + [logit(z[n]) for n in own_names])
d = X.shape[1]

GROUPS = [
    ["xgb_v3", "xgb_v3_s1", "xgb_v3_d10"],
    ["realmlp_pub", "realmlp_pub_s1", "realmlp_pub_s2"],
    ["realmlp_v4", "realmlp_v4_s1", "realmlp_v4_s2"],
    ["realmlp_fa", "realmlp_fa_s7", "realmlp_fa_s123"],
    ["cat_ctr", "cat_ctr3", "cat_ctr5"],
    ["realmlp_fa_aux", "realmlp_fa_aux_s2"],
    ["realmlp_vb_aux", "realmlp_vb_aux_s2"],
    ["tabpfn_full_raw", "tabpfn_full_raw_s1"],
    ["tabpfn_catfd", "tabpfn_catfd_s1"],
]
name2idx = {n: i for i, n in enumerate(names)}
A = np.zeros((d, d))
for g in GROUPS:
    idx = [name2idx[n] for n in g if n in name2idx]
    for a in idx:
        for b in idx:
            if a != b:
                A[a, b] = 1.0
L = np.diag(A.sum(axis=1)) - A


def fit_grouped(Xtr, ytr, lam=1.0, C=1.0):
    ypm = (2 * ytr - 1).astype(np.float64)

    def obj(theta):
        w = theta[:d]; c = theta[d]
        s = Xtr @ w + c
        logloss = np.logaddexp(0, -ypm * s)
        loss = 0.5 * (w @ w) + C * logloss.sum() + lam * (w @ (L @ w))
        sig = 1.0 / (1.0 + np.exp(ypm * s))
        grad_w = w - C * (Xtr.T @ (ypm * sig)) + 2 * lam * (L @ w)
        grad_c = -C * (ypm * sig).sum()
        return loss, np.concatenate([grad_w, [grad_c]])

    res = minimize(obj, np.zeros(d + 1), jac=True, method="L-BFGS-B", options={"maxiter": 500, "ftol": 1e-12})
    return res.x[:d], res.x[d]


FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=33).split(X, y))
ys, lrs, grps = [], [], []
for k, (ia, iv) in enumerate(FOLDS[:2]):
    m = LogisticRegression(max_iter=2000, C=1.0).fit(X[ia], y[ia])
    oof_lr = m.decision_function(X[iv])
    w, c = fit_grouped(X[ia], y[ia])
    oof_grp = X[iv] @ w + c
    a_lr, a_grp = roc_auc_score(y[iv], oof_lr), roc_auc_score(y[iv], oof_grp)
    print(f"fold {k}: lr {a_lr:.6f}  grouped {a_grp:.6f}  Δ {a_grp-a_lr:+.6f}", flush=True)
    ys.append(y[iv]); lrs.append(oof_lr); grps.append(oof_grp)

y_all = np.concatenate(ys); lr_all = np.concatenate(lrs); grp_all = np.concatenate(grps)
print(f"\n合并两折: lr {roc_auc_score(y_all, lr_all):.6f}  grouped {roc_auc_score(y_all, grp_all):.6f}", flush=True)
_, _, delta, lo, hi, p = delong_ci(y_all, lr_all, grp_all)
print(f"配对 DeLong: Δ={delta:+.7f}  95%CI=[{lo:+.7f}, {hi:+.7f}]  p={p:.4f}", flush=True)
