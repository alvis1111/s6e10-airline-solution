"""S6E10 — 三个固定对照，核验自有模型加入 Naji 后是否仍有稳定贡献。

1. 100% Naji02（不掺自有模型）
2. 当前融合权重（full-OOF 坐标下降所得，提交 LB 0.96155 的权重）
3. 独立划分定权重的诚实融合：外层 5 折，每折在 train 折上坐标下降定权重、val 折评估

关键问题：自有模型(route_aux/tabm/tabpfn)在 Naji 之外是否还稳定贡献？若诚实 OOF 不显著高于
Naji02 单腿，说明融合收益是权重搜索在 full-OOF 上的过拟合。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv('data/playground-series-s6e10/train.csv')
test = pd.read_csv('data/playground-series-s6e10/test.csv')
sub = pd.read_csv('data/playground-series-s6e10/sample_submission.csv')
y = train['satisfaction'].to_numpy(np.int64)
def r(p): return rankdata(p)/len(p)

route_aux_oof = 0.5*np.load('cache/oof_route_aux_l.npy')+0.5*np.load('cache/oof_route_aux_x.npy')
tabm_oof = np.load('cache/oof_route_tabm.npy')
TABPFN = ['tabpfn_te4','tabpfn_te','tabpfn_catfd10','tabpfn_catfd']
tab_oof = [pd.read_csv(f'pulled_tabpfn/oof_{m}.csv')['pred'].to_numpy() for m in TABPFN]
naji01_oof = pd.read_csv('pulled_naji/01_blend_oof.csv')['satisfaction'].to_numpy()
naji02_oof = pd.read_csv('pulled_naji/02_blend_oof.csv')['satisfaction'].to_numpy()
names = ['route_aux','tabm','tabpfn_te4','tabpfn_te','tabpfn_catfd10','tabpfn_catfd','naji02','naji01']
oof_list = [route_aux_oof, tabm_oof] + tab_oof + [naji02_oof, naji01_oof]
R = np.column_stack([r(p) for p in oof_list])

# 1. 100% Naji02
print(f"1. 100% Naji02           OOF = {roc_auc_score(y, naji02_oof):.5f}")

# 2. 当前融合权重（full-OOF 坐标下降）
w_cur = np.array([0.188, 0.0, 0.0, 0.0, 0.063, 0.031, 0.624, 0.094])
w_cur = w_cur/w_cur.sum()
print(f"2. 当前融合(full-OOF权重) OOF = {roc_auc_score(y, R@w_cur):.5f}")

# 3. 独立划分定权重：外层5折，train折坐标下降(快速)，val折评估
def fast_coord(Rtr, ytr, iters=6, grid=9):
    k = Rtr.shape[1]; w = np.ones(k)/k
    def auc(w): return roc_auc_score(ytr, Rtr @ (w/w.sum()))
    for _ in range(iters):
        for i in range(k):
            best = (auc(w), w[i])
            for wi in np.linspace(0,1,grid):
                wt = w.copy(); wt[i]=wi; wt=wt/wt.sum()
                a = auc(wt)
                if a > best[0]: best = (a, wi)
            w[i] = best[1]
    return w/w.sum()

outer = StratifiedKFold(5, shuffle=True, random_state=7)
oof_honest = np.zeros(len(y)); ws = []
for tr, va in outer.split(R, y):
    w = fast_coord(R[tr], y[tr]); ws.append(w)
    oof_honest[va] = R[va] @ w
print(f"3. 独立划分定权重(诚实)  OOF = {roc_auc_score(y, oof_honest):.5f}")
print("   各折权重均值:", dict(zip(names, np.round(np.mean(ws, axis=0), 3))))
print("   逐折权重:")
for i, w in enumerate(ws):
    nnz = [(n, round(x,2)) for n,x in zip(names,w) if x > 0.01]
    print(f"     fold {i}: {nnz}")
