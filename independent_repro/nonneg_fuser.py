"""非负权重融合对照：原 Busy53(无约束 logistic) vs 相同 53 成员 + 非负系数约束。

同一 5 折 seed 33、logit 输入、C=1.0。候选 = 同一 log-loss + L2 目标，但 w≥0（截距自由）。
先跑两折看门，再决定是否补五折（这里直接跑五折看全貌）。
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
X = np.column_stack([logit(mo[c].to_numpy()) for c in mo.columns] + [logit(z[n]) for n in own_names])
print(f"53 成员: {X.shape}", flush=True)


def fit_nonneg(Xtr, ytr, C=1.0):
    """非负 logistic 回归：目标 0.5*||w||² + C*Σlog(1+exp(-y*s))，w≥0、截距自由。"""
    n, d = Xtr.shape
    ypm = (2 * ytr - 1).astype(np.float64)          # {-1, +1}

    def obj(theta):
        w = theta[:d]; c = theta[d]
        s = Xtr @ w + c
        logloss = np.logaddexp(0, -ypm * s)          # log(1+exp(-y*s))
        loss = 0.5 * (w @ w) + C * logloss.sum()
        sig = 1.0 / (1.0 + np.exp(ypm * s))          # σ(-y*s)
        grad_w = w - C * (Xtr.T @ (ypm * sig))
        grad_c = -C * (ypm * sig).sum()
        return loss, np.concatenate([grad_w, [grad_c]])

    bounds = [(0, None)] * d + [(None, None)]
    res = minimize(obj, np.zeros(d + 1), jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500, "ftol": 1e-12})
    return res.x[:d], res.x[d]


def predict(w, c, X):
    return X @ w + c


FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=33).split(X, y))
oof_lr = np.zeros(len(y)); oof_nn = np.zeros(len(y))
neg_counts = []
for k, (ia, iv) in enumerate(FOLDS):
    m = LogisticRegression(max_iter=2000, C=1.0).fit(X[ia], y[ia])
    oof_lr[iv] = m.decision_function(X[iv])
    neg_counts.append(int((m.coef_[0] < 0).sum()))
    w, c = fit_nonneg(X[ia], y[ia])
    oof_nn[iv] = predict(w, c, X[iv])
    print(f"fold {k}: lr {roc_auc_score(y[iv], oof_lr[iv]):.6f}  nonneg {roc_auc_score(y[iv], oof_nn[iv]):.6f}  "
          f"(负系数 {neg_counts[-1]}/53)", flush=True)

print(f"\n无约束 logistic OOF = {roc_auc_score(y, oof_lr):.6f}", flush=True)
print(f"非负 logistic   OOF = {roc_auc_score(y, oof_nn):.6f}", flush=True)
print(f"各折负系数数: {neg_counts}", flush=True)
_, _, delta, lo, hi, p = delong_ci(y, oof_lr, oof_nn)
print(f"配对 DeLong: Δ={delta:+.7f}  95%CI=[{lo:+.7f}, {hi:+.7f}]  p={p:.4f}", flush=True)
