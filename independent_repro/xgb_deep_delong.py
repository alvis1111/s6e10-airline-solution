"""深层 XGBoost 两折（fold 4/5）配对 DeLong：50/50 基线 vs +10% 深层XGB。"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from delong import delong_ci

train = pd.read_csv("data/playground-series-s6e10/train.csv")
y = train["satisfaction"].to_numpy(np.int64)

route_aux = 0.5 * np.load("cache/oof_route_aux_l.npy") + 0.5 * np.load("cache/oof_route_aux_x.npy")
z = np.load("digit_confirm_results/refined_components.npz", allow_pickle=False)
order = {tid: i for i, tid in enumerate(train["id"].to_numpy())}
ctx = np.full(len(train), np.nan); ctx[[order[t] for t in z["train_id"]]] = z["context_oof"]
tabm = np.full(len(train), np.nan); tabm[[order[t] for t in z["train_id"]]] = z["route_tabm_oof"]
rmlp = np.load("cache/oof_realmlp_lean.npy")
rk = lambda a: rankdata(a) / len(a)
fusion = 0.3 * rk(route_aux) + 0.3 * rk(ctx) + 0.4 * rk(tabm)
blend_5050 = 0.5 * rk(fusion) + 0.5 * rk(rmlp)

ys, bases, news = [], [], []
for fold in [4, 5]:
    d = np.load(f"cache/xgb_deep_fold{fold}.npz")
    pos, yv, pred = d["val_id"], d["y_val"], d["pred"]
    base = blend_5050[pos]
    new = 0.9 * rk(base) + 0.1 * rk(pred)
    print(f"fold {fold}: 50/50 {roc_auc_score(yv, base):.6f} -> +10% {roc_auc_score(yv, new):.6f}  "
          f"Δ {roc_auc_score(yv, new) - roc_auc_score(yv, base):+.6f}", flush=True)
    ys.append(yv); bases.append(base); news.append(new)

y_all = np.concatenate(ys); b_all = np.concatenate(bases); n_all = np.concatenate(news)
print(f"\n合并两折 {len(y_all)} 行: 50/50 {roc_auc_score(y_all, b_all):.6f}  +10% {roc_auc_score(y_all, n_all):.6f}", flush=True)
_, _, delta, lo, hi, p = delong_ci(y_all, b_all, n_all)
print(f"配对 DeLong: Δ={delta:+.7f}  95%CI=[{lo:+.7f}, {hi:+.7f}]  p={p:.4f}", flush=True)
