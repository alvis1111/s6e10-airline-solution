"""S6E10 — 正确归因：自有模型(route_aux+tabm)在公开预测(Naji+TabPFN)之外的边际贡献。

比较两个集合的诚实嵌套 CV：
  A. 完整 8 腿 = 4 TabPFN + naji02 + naji01 + route_aux + tabm
  B. 纯公开 6 腿 = 4 TabPFN + naji02 + naji01（去掉自有模型，重新定权重）
  C. 100% naji02（参照）

A-B = 自有模型(route_aux+tabm)的边际贡献。注意：这里的"嵌套"仍是「对已有 OOF 再划折」，
不是完整嵌套验证，结论只能算「本次划分下的观察」。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv('data/playground-series-s6e10/train.csv')
y = train['satisfaction'].to_numpy(np.int64)
def r(p): return rankdata(p)/len(p)

route_aux = 0.5*np.load('cache/oof_route_aux_l.npy')+0.5*np.load('cache/oof_route_aux_x.npy')
tabm = np.load('cache/oof_route_tabm.npy')
TABPFN = ['tabpfn_te4','tabpfn_te','tabpfn_catfd10','tabpfn_catfd']
tab = [pd.read_csv(f'pulled_tabpfn/oof_{m}.csv')['pred'].to_numpy() for m in TABPFN]
naji01 = pd.read_csv('pulled_naji/01_blend_oof.csv')['satisfaction'].to_numpy()
naji02 = pd.read_csv('pulled_naji/02_blend_oof.csv')['satisfaction'].to_numpy()

full_names = ['route_aux','tabm','tabpfn_te4','tabpfn_te','tabpfn_catfd10','tabpfn_catfd','naji02','naji01']
full_legs = [route_aux, tabm] + tab + [naji02, naji01]
pub_names = ['tabpfn_te4','tabpfn_te','tabpfn_catfd10','tabpfn_catfd','naji02','naji01']
pub_legs = tab + [naji02, naji01]

def rmat(legs): return np.column_stack([r(p) for p in legs])

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

def nested_oof(legs):
    R = rmat(legs); outer = StratifiedKFold(5, shuffle=True, random_state=7)
    oof = np.zeros(len(y)); ws = []
    for tr, va in outer.split(R, y):
        w = fast_coord(R[tr], y[tr]); ws.append(w)
        oof[va] = R[va] @ w
    return roc_auc_score(y, oof), np.mean(ws, axis=0)

print(f"C. 100% naji02               OOF = {roc_auc_score(y, naji02):.5f}")
a, wa = nested_oof(full_legs)
print(f"A. 完整 8 腿                 OOF = {a:.5f}")
print("   权重均值:", dict(zip(full_names, np.round(wa,3))))
b, wb = nested_oof(pub_legs)
print(f"B. 纯公开 6 腿(去自有)        OOF = {b:.5f}")
print("   权重均值:", dict(zip(pub_names, np.round(wb,3))))
print(f"\n自有模型(route_aux+tabm)边际贡献 = {a-b:+.5f}  (A-B)")
