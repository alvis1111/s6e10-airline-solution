"""深层 XGBoost 两折（fold 4/5，修正内层ES后）配对 DeLong：0.96125 融合 vs +10% 深层XGB。"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from delong import delong_ci

train = pd.read_csv("data/playground-series-s6e10/train.csv")
y = train["satisfaction"].to_numpy(np.int64)
base_z = np.load(r"C:/Users/GEM07/Documents/Codex/2026-10-02/https-www-kaggle-com-competitions-playground/outputs/lean_aux_confirmation/full_ed71a6690aa6/predictions.npz", allow_pickle=False)
order = {tid: i for i, tid in enumerate(train["id"].to_numpy())}
oof_cand = np.full(len(train), np.nan)
oof_cand[[order[t] for t in base_z["train_id"]]] = base_z["oof_candidate"]
assert not np.isnan(oof_cand).any()
rk = lambda a: rankdata(a) / len(a)

ys, bases, news = [], [], []
for fold in [4, 5]:
    d = np.load(f"cache/xgb_deep_fold{fold}.npz")
    pos, yv, pred = d["val_id"], d["y_val"], d["pred"]
    base = oof_cand[pos]
    new = 0.9 * rk(base) + 0.1 * rk(pred)
    a_b, a_n = roc_auc_score(yv, base), roc_auc_score(yv, new)
    print(f"fold {fold}: 0.96125 {a_b:.7f} -> +10% {a_n:.7f}  Δ {a_n - a_b:+.7f}", flush=True)
    ys.append(yv); bases.append(base); news.append(new)

y_all = np.concatenate(ys); b_all = np.concatenate(bases); n_all = np.concatenate(news)
print(f"\n合并两折 {len(y_all)} 行: 0.96125 {roc_auc_score(y_all, b_all):.7f}  +10% {roc_auc_score(y_all, n_all):.7f}", flush=True)
_, _, delta, lo, hi, p = delong_ci(y_all, b_all, n_all)
print(f"配对 DeLong: Δ={delta:+.7f}  95%CI=[{lo:+.7f}, {hi:+.7f}]  p={p:.4f}", flush=True)
