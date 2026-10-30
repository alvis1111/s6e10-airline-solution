"""S6E10 — 把公开 TabPFN 预测混进最终融合，生成提交。

腿：route+aux(10折3种子) + context + tabm + 4个 TabPFN 成员。
用受约束凸组合（w>=0, sum=1, log-loss）+ 嵌套 CV 得诚实 OOF，再全量拟合生成提交。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv('data/playground-series-s6e10/train.csv')
test = pd.read_csv('data/playground-series-s6e10/test.csv')
sub = pd.read_csv('data/playground-series-s6e10/sample_submission.csv')
y = train['satisfaction'].to_numpy(np.int64)

TABPFN = ['tabpfn_te4', 'tabpfn_te', 'tabpfn_catfd10', 'tabpfn_catfd']
oof, pred = {}, {}
# route + aux（10折3种子 LGBM+XGB blend）
oof['route_aux'] = 0.5*np.load('cache/oof_route_aux_l.npy') + 0.5*np.load('cache/oof_route_aux_x.npy')
pred['route_aux'] = 0.5*np.load('cache/pred_route_aux_l.npy') + 0.5*np.load('cache/pred_route_aux_x.npy')
# context
z = np.load(r'C:/Users/GEM07/OneDrive/Desktop/s6e10_outer10_blend/previous_context_ensemble.npz')
oof['context'] = z['oof']; pred['context'] = z['pred']
# tabm
oof['tabm'] = np.load('cache/oof_route_tabm.npy'); pred['tabm'] = np.load('cache/pred_route_tabm.npy')
# tabpfn
for m in TABPFN:
    oof[m] = pd.read_csv(f'pulled_tabpfn/oof_{m}.csv')['pred'].to_numpy()
    pred[m] = pd.read_csv(f'pulled_tabpfn/test_{m}.csv')['pred'].to_numpy()

names = list(oof.keys())
X = np.column_stack([oof[n] for n in names])
XT = np.column_stack([pred[n] for n in names])

print('=== 单腿 OOF ===')
for n in names:
    print(f'  {n:16s} {roc_auc_score(y, oof[n]):.5f}')

def fit_convex(Xt, yt):
    def negll(w):
        p = np.clip(Xt @ w, 1e-6, 1-1e-6)
        return -np.mean(yt*np.log(p) + (1-yt)*np.log(1-p))
    return minimize(negll, np.full(Xt.shape[1], 1/Xt.shape[1]), method='SLSQP',
                    bounds=[(0,1)]*Xt.shape[1],
                    constraints={'type':'eq','fun':lambda w:w.sum()-1}).x

# 嵌套 CV 诚实 OOF
outer = StratifiedKFold(5, shuffle=True, random_state=7)
oof_stack = np.zeros(len(y))
ws = []
for tr, va in outer.split(X, y):
    w = fit_convex(X[tr], y[tr]); ws.append(w)
    oof_stack[va] = X[va] @ w
print(f'\n=== 凸组合(嵌套CV) 诚实 OOF = {roc_auc_score(y, oof_stack):.5f} ===')

w_full = fit_convex(X, y)
print('\n全量拟合权重:')
for n, w in zip(names, w_full):
    print(f'  {n:16s} {w:.4f}')

# 生成提交
sub['satisfaction'] = XT @ w_full
sub.to_csv('submission_tabpfn_blend.csv', index=False)
print('\nsaved -> submission_tabpfn_blend.csv')
