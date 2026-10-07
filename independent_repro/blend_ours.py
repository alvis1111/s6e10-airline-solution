"""blend_ours.py —— 自有模型多腿网格搜索（5 路 vs 4 路去掉 realmlp）。

把 route_aux 拆成 LGBM/XGBoost 两腿 + context + tabm + realmlp(lean) 共 5 腿，
rank-blend 网格搜索（0.1 步粗搜 → 坐标下降精搜），打印最优权重与 OOF。
不再手调 30/30/40。⚠️ 在 full-OOF 上搜权会高估（见 memory「OOF 搜权高估」），结果作参考。
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

train = pd.read_csv("data/playground-series-s6e10/train.csv")
y = train["satisfaction"].to_numpy(np.int64)
n = len(train)
rk = lambda a: rankdata(a) / len(a)

route_l = np.load("cache/oof_route_aux_l.npy")
route_x = np.load("cache/oof_route_aux_x.npy")
z = np.load("digit_confirm_results/refined_components.npz", allow_pickle=False)
order = {tid: i for i, tid in enumerate(train["id"].to_numpy())}
ctx = np.full(n, np.nan); ctx[[order[t] for t in z["train_id"]]] = z["context_oof"]
tabm = np.full(n, np.nan); tabm[[order[t] for t in z["train_id"]]] = z["route_tabm_oof"]
rmlp = np.load("cache/oof_realmlp_lean.npy")
assert not np.isnan(ctx).any() and not np.isnan(tabm).any()

LEGS = {
    "route_lgb": rk(route_l),
    "route_xgb": rk(route_x),
    "context": rk(ctx),
    "tabm": rk(tabm),
    "realmlp": rk(rmlp),
}
names = list(LEGS.keys())
R = np.column_stack([LEGS[k] for k in names])


def auc_of(w):
    return roc_auc_score(y, R @ (w / w.sum()))


def coord(w, iters=12, grid=21):
    k = len(w)
    w = np.array(w, dtype=float)
    for _ in range(iters):
        for i in range(k):
            best = (auc_of(w), w[i])
            for wi in np.linspace(0, 1, grid):
                wt = w.copy(); wt[i] = wi
                a = auc_of(wt)
                if a > best[0]:
                    best = (a, wi)
            w[i] = best[1]
    return w / w.sum()


def coarse_grid():
    """0.1 步长的 5 维权重粗搜（组合数 C(14,4)=1001）。"""
    best = (0, None)
    for w1 in range(11):
        for w2 in range(11 - w1):
            for w3 in range(11 - w1 - w2):
                for w4 in range(11 - w1 - w2 - w3):
                    w5 = 10 - w1 - w2 - w3 - w4
                    w = np.array([w1, w2, w3, w4, w5], dtype=float) / 10
                    a = auc_of(w)
                    if a > best[0]:
                        best = (a, w)
    return best


def search(indices):
    """对指定腿的子集做粗搜 + 精搜，返回 (OOF, 全腿权重)。"""
    sub = R[:, indices]
    def sub_auc(w):
        return roc_auc_score(y, sub @ (w / w.sum()))
    # 粗搜
    best = (0, None)
    for w1 in range(11):
        for w2 in range(11 - w1):
            for w3 in range(11 - w1 - w2):
                if len(indices) == 4:
                    w4 = 10 - w1 - w2 - w3
                    w = np.array([w1, w2, w3, w4], dtype=float) / 10
                else:  # 5 腿
                    for w4 in range(11 - w1 - w2 - w3):
                        w5 = 10 - w1 - w2 - w3 - w4
                        w = np.array([w1, w2, w3, w4, w5], dtype=float) / 10
                        a = sub_auc(w)
                        if a > best[0]:
                            best = (a, w)
                    continue
                a = sub_auc(w)
                if a > best[0]:
                    best = (a, w)
    # 精搜（坐标下降）
    w = best[1]
    for _ in range(12):
        for i in range(len(indices)):
            cur = (sub_auc(w), w[i])
            for wi in np.linspace(0, 1, 21):
                wt = w.copy(); wt[i] = wi
                a = sub_auc(wt)
                if a > cur[0]:
                    cur = (a, wi)
            w[i] = cur[1]
    w = w / w.sum()
    full = np.zeros(len(names))
    full[indices] = w
    return sub_auc(w), full


print("=== 5 路（route_lgb + route_xgb + context + tabm + realmlp） ===", flush=True)
oof5, w5 = search(list(range(5)))
print(f"OOF = {oof5:.6f}")
for nm, w in zip(names, w5):
    print(f"  {nm:10s} {w:.4f}")

print("\n=== 4 路（去掉 realmlp） ===", flush=True)
oof4, w4 = search([0, 1, 2, 3])
print(f"OOF = {oof4:.6f}")
for nm, w in zip(names, w4):
    print(f"  {nm:10s} {w:.4f}")

print(f"\nrealmlp 边际（5 路 vs 4 路）= {oof5 - oof4:+.6f}", flush=True)
