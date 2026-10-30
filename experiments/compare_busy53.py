"""本地复现 compare_busy53_naji.ipynb：Busyaprime 53 与 Naji02 对照 + 混合筛选。"""
import numpy as np
import pandas as pd
import json
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

train = pd.read_csv('data/playground-series-s6e10/train.csv', usecols=['id','satisfaction'])
test = pd.read_csv('data/playground-series-s6e10/test.csv', usecols=['id'])
y = train.satisfaction.to_numpy(dtype=np.int64)

naji = pd.read_csv('pulled_naji/02_blend_oof.csv').set_index('id').loc[train.id,'satisfaction'].to_numpy()
naji_test = pd.read_csv('pulled_naji/02_submission.csv').set_index('id').loc[test.id,'satisfaction'].to_numpy()
assert abs(roc_auc_score(y, naji) - 0.9619203392791454) < 1e-8

with np.load('pulled_busy53/oof_stacks.npz', allow_pickle=False) as z:
    oof48 = np.array(z['oof48']); oof53 = np.array(z['oof53']); test53 = np.array(z['test53'])
assert abs(roc_auc_score(y, oof48) - 0.961919) < 5e-7, '行序不一致'
busy_logits = oof53.astype(np.float64); busy_test_logits = test53.astype(np.float64)

baseline_auc = float(roc_auc_score(y, naji))
busy_auc = float(roc_auc_score(y, busy_logits))
print(f'Naji02 OOF: {baseline_auc:.9f}')
print(f'Busy53 OOF: {busy_auc:.9f}; delta {busy_auc-baseline_auc:+.9f}')

# 导出原版 Busy53 提交（供用户做原版对照）
busy_sub = pd.read_csv('pulled_busy53/submission.csv')
busy_sub.to_csv('submission_busy53.csv', index=False)
print('已导出 submission_busy53.csv（原版 Busy53，可直接提交）')

# 混合筛选（discovery/confirmation + 配对 DeLong）
plan_seed = 20261007
weights = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
discovery, confirmation = train_test_split(np.arange(len(y)), test_size=0.5, stratify=y, random_state=plan_seed)
naji_logits = logit(np.clip(naji, 1e-6, 1-1e-6))

rows = []
for w in weights:
    v = naji if w == 0 else busy_logits if w == 1 else (1-w)*naji_logits + w*busy_logits
    rows.append({'busy_weight': w, 'discovery_auc': float(roc_auc_score(y[discovery], v[discovery]))})
chosen = max(rows, key=lambda r: r['discovery_auc'])
w = chosen['busy_weight']
print(f'\ndiscovery 选中的权重 w={w}')
for r in rows:
    print(f"  w={r['busy_weight']}  discovery_auc={r['discovery_auc']:.9f}")

chosen_logits = naji if w == 0 else busy_logits if w == 1 else (1-w)*naji_logits + w*busy_logits

def placements(target, prediction):
    pos, neg = prediction[target == 1], prediction[target == 0]
    sp, sn = np.sort(pos), np.sort(neg)
    vp = (np.searchsorted(sn, pos, side='left') + np.searchsorted(sn, pos, side='right')) / (2.0*len(neg))
    vn = 1 - (np.searchsorted(sp, neg, side='left') + np.searchsorted(sp, neg, side='right')) / (2.0*len(pos))
    return vp, vn

def paired_delta(target, reference, candidate):
    ap, an = placements(target, reference)
    bp, bn = placements(target, candidate)
    delta = float((bp-ap).mean())
    se = float(np.sqrt(np.var(bp-ap, ddof=1)/len(ap) + np.var(bn-an, ddof=1)/len(an)))
    return {'delta': delta, 'ci95': [delta-1.95996398454*se, delta+1.95996398454*se]}

d_naji = paired_delta(y[confirmation], naji[confirmation], chosen_logits[confirmation])
d_busy = paired_delta(y[confirmation], busy_logits[confirmation], chosen_logits[confirmation])
passed = 0 < w < 1 and all(d['delta'] >= 0.00002 and d['ci95'][0] > 0 for d in [d_naji, d_busy])
print(f'\nconfirmation delta vs Naji: {d_naji}')
print(f'confirmation delta vs Busy53: {d_busy}')
print(f'blend gate passed: {passed}')
if passed:
    pred = expit((1-w)*logit(np.clip(naji_test,1e-6,1-1e-6)) + w*busy_test_logits)
    pd.DataFrame({'id': test.id, 'satisfaction': pred}).to_csv('submission_validated_blend.csv', index=False)
    print('已导出 submission_validated_blend.csv')
else:
    print('未通过筛选，保留纯 Busy53 提交')
