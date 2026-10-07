"""RealMLP 精简版两折配对 DeLong：原融合 vs 原融合+10%RealMLP（rank 融合）。"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from delong import delong_ci

train = pd.read_csv("data/playground-series-s6e10/train.csv")
y = train["satisfaction"].to_numpy(np.int64)
n = len(train)

route_aux = 0.5 * np.load("cache/oof_route_aux_l.npy") + 0.5 * np.load("cache/oof_route_aux_x.npy")
z = np.load("digit_confirm_results/refined_components.npz", allow_pickle=False)
order = {tid: i for i, tid in enumerate(train["id"].to_numpy())}
ctx = np.full(n, np.nan); ctx[[order[t] for t in z["train_id"]]] = z["context_oof"]
tabm = np.full(n, np.nan); tabm[[order[t] for t in z["train_id"]]] = z["route_tabm_oof"]
assert not np.isnan(ctx).any() and not np.isnan(tabm).any()

rk = lambda a: rankdata(a) / len(a)
fusion = 0.3 * rk(route_aux) + 0.3 * rk(ctx) + 0.4 * rk(tabm)

ys, bases, news = [], [], []
for fold in [0, 1]:
    d = np.load(f"cache/realmlp_lean_fold{fold}.npz")
    pos = d["val_id"]                       # = 位置 iv（train id 即行号）
    yv = d["y_val"]; pred = d["pred"]
    base = fusion[pos]
    new = 0.9 * rk(base) + 0.1 * rk(pred)
    print(f"fold {fold}: 融合 {roc_auc_score(yv, base):.6f} -> +10%RealMLP {roc_auc_score(yv, new):.6f}  Δ {roc_auc_score(yv,new)-roc_auc_score(yv,base):+.6f}", flush=True)
    ys.append(yv); bases.append(base); news.append(new)

y_all = np.concatenate(ys); base_all = np.concatenate(bases); new_all = np.concatenate(news)
print(f"\n合并两折 {len(y_all)} 行", flush=True)
print(f"融合 AUC {roc_auc_score(y_all, base_all):.6f}  +10%RealMLP AUC {roc_auc_score(y_all, new_all):.6f}", flush=True)
auc_b, auc_n, delta, lo, hi, p = delong_ci(y_all, base_all, new_all)
print(f"配对 DeLong: Δ={delta:+.7f}  95%CI=[{lo:+.7f}, {hi:+.7f}]  p={p:.4f}", flush=True)
