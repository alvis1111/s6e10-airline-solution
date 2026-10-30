"""90% Busy53 + 10% 自有 0.96126 融合，rank 加权。OOF 先估算 + 生成提交。"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

train = pd.read_csv("data/playground-series-s6e10/train.csv")
test = pd.read_csv("data/playground-series-s6e10/test.csv")
y = train["satisfaction"].to_numpy(np.int64)
rk = lambda a: rankdata(a) / len(a)

# --- OOF 估算 ---
oof53 = np.load("pulled_busy53/oof_stacks.npz")["oof53"]
print(f"Busy53 OOF AUC = {roc_auc_score(y, oof53):.6f}")

base_z = np.load(r"C:/Users/GEM07/Documents/Codex/2026-10-02/https-www-kaggle-com-competitions-playground/outputs/deep_xgb_confirmation/full_096126/predictions.npz", allow_pickle=False)
order = {tid: i for i, tid in enumerate(train["id"].to_numpy())}
oof_cand = np.full(len(train), np.nan); oof_cand[[order[t] for t in base_z["train_id"]]] = base_z["oof_candidate"]
oof_xgb = np.load("cache/oof_xgb_deep.npy")
own_oof = 0.9 * rk(oof_cand) + 0.1 * rk(oof_xgb)
print(f"自有 0.96126 OOF AUC = {roc_auc_score(y, own_oof):.6f}")

blend_oof = 0.9 * rk(oof53) + 0.1 * rk(own_oof)
print(f"90/10 融合 OOF = {roc_auc_score(y, blend_oof):.6f}  Δ vs 纯 Busy53 = {roc_auc_score(y, blend_oof) - roc_auc_score(y, oof53):+.6f}")

# --- 生成提交 ---
busy_te = pd.read_csv("pulled_busy53/submission.csv").set_index("id").reindex(test["id"])["satisfaction"].to_numpy()
own_te = pd.read_csv("submission_xgb_deep.csv").set_index("id").reindex(test["id"])["satisfaction"].to_numpy()
blend_te = 0.9 * rk(busy_te) + 0.1 * rk(own_te)

sub = pd.read_csv("data/playground-series-s6e10/sample_submission.csv")
sub["satisfaction"] = blend_te
sub.to_csv("submission_busy53_90_ours_10.csv", index=False)
print("saved -> submission_busy53_90_ours_10.csv")
