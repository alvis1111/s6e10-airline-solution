"""AUC 导向线性融合器 vs logistic 回归（53 成员）。

复现 Busy 原版：53 成员（48 公开 + 5 自有）概率→logit，logistic 回归(log-loss)嵌套 5 折 seed 33。
对比：同样 53 个 logit 特征上，改成对排序损失（pairwise logistic loss + L2）训练的线性融合器。
固定 5 折 seed 33，OOF + 配对 DeLong 比较。
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from delong import delong_ci

train = pd.read_csv("data/playground-series-s6e10/train.csv")
y = train["satisfaction"].to_numpy(np.int64)

MEM = r"C:/Users/GEM07/.cache/kagglehub/notebooks/goodpjw2008/s6e10-tabpfn-route-categories-lb-0-96160/output/versions/1"
mo = pd.read_csv(MEM + "/oof_members.csv").set_index("id").reindex(train["id"])  # 48 公开成员（概率）
z = np.load("pulled_busy53/oof_ours.npz", allow_pickle=False)  # 5 自有成员（概率）
own_names = ["lgbm", "lgbm_xt", "cat", "xgb", "realmlp"]

logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
X = np.column_stack([logit(mo[c].to_numpy()) for c in mo.columns] + [logit(z[n]) for n in own_names])
print(f"53 成员特征 X: {X.shape}", flush=True)

FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=33).split(X, y))


def fit_pairwise(Xtr, ytr, l2=0.01, n_pairs=2_000_000, seed=0):
    """在固定采样对上最小化成对 logistic loss + L2（L-BFGS），返回线性权重 w。"""
    rng = np.random.RandomState(seed)
    pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
    pi = rng.choice(pos, n_pairs, replace=True)
    nj = rng.choice(neg, n_pairs, replace=True)
    Xp = Xtr[pi]; Xn = Xtr[nj]

    def loss_grad(w):
        d = (Xp - Xn) @ w                       # pos - neg 得分差
        sig = 1.0 / (1.0 + np.exp(d))           # σ(-d)
        loss = float(np.log(1 + np.exp(-d)).mean() + l2 * (w @ w))
        grad = -(sig[:, None] * (Xp - Xn)).mean(axis=0) + 2 * l2 * w
        return loss, grad

    res = minimize(loss_grad, np.zeros(Xtr.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 200, "ftol": 1e-9})
    return res.x


oof_lr = np.zeros(len(y))
oof_pw = np.zeros(len(y))
for k, (ia, iv) in enumerate(FOLDS):
    m = LogisticRegression(max_iter=2000).fit(X[ia], y[ia])
    oof_lr[iv] = m.decision_function(X[iv])
    w = fit_pairwise(X[ia], y[ia])
    oof_pw[iv] = X[iv] @ w
    print(f"fold {k}: lr {roc_auc_score(y[iv], oof_lr[iv]):.6f}  pairwise {roc_auc_score(y[iv], oof_pw[iv]):.6f}", flush=True)

print(f"\nlogistic(复现) OOF = {roc_auc_score(y, oof_lr):.6f}", flush=True)
print(f"pairwise 融合器 OOF = {roc_auc_score(y, oof_pw):.6f}", flush=True)
_, _, delta, lo, hi, p = delong_ci(y, oof_lr, oof_pw)
print(f"配对 DeLong: Δ={delta:+.7f}  95%CI=[{lo:+.7f}, {hi:+.7f}]  p={p:.4f}", flush=True)
np.save("cache/oof_rank_fuser.npy", oof_pw)
