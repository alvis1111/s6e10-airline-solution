"""按 50/50 重新生成提交：0.5·rank(融合) + 0.5·rank(RealMLP)。"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

train = pd.read_csv("data/playground-series-s6e10/train.csv")
test = pd.read_csv("data/playground-series-s6e10/test.csv")
sub = pd.read_csv("data/playground-series-s6e10/sample_submission.csv")
y = train["satisfaction"].to_numpy(np.int64)
rk = lambda a: rankdata(a) / len(a)

route_aux = 0.5 * np.load("cache/oof_route_aux_l.npy") + 0.5 * np.load("cache/oof_route_aux_x.npy")
route_aux_te = 0.5 * np.load("cache/pred_route_aux_l.npy") + 0.5 * np.load("cache/pred_route_aux_x.npy")
z = np.load("digit_confirm_results/refined_components.npz", allow_pickle=False)
tr_ord = {tid: i for i, tid in enumerate(train["id"].to_numpy())}
ctx = np.full(len(train), np.nan); ctx[[tr_ord[t] for t in z["train_id"]]] = z["context_oof"]
tabm = np.full(len(train), np.nan); tabm[[tr_ord[t] for t in z["train_id"]]] = z["route_tabm_oof"]
ctx_z = np.load("C:/Users/GEM07/OneDrive/Desktop/s6e10_outer10_blend/previous_context_ensemble.npz", allow_pickle=False)
te_ord = {tid: i for i, tid in enumerate(test["id"].to_numpy())}
ctx_te = np.full(len(test), np.nan); ctx_te[[te_ord[t] for t in ctx_z["test_id"]]] = ctx_z["pred"]
tabm_te = np.load("cache/pred_route_tabm.npy")
assert not np.isnan(ctx).any() and not np.isnan(tabm).any() and not np.isnan(ctx_te).any()

fusion_oof = 0.3 * rk(route_aux) + 0.3 * rk(ctx) + 0.4 * rk(tabm)
fusion_te = 0.3 * rk(route_aux_te) + 0.3 * rk(ctx_te) + 0.4 * rk(tabm_te)
rmlp_oof = np.load("cache/oof_realmlp_lean.npy")
rmlp_te = np.load("cache/pred_realmlp_lean.npy")

blend_oof = 0.5 * rk(fusion_oof) + 0.5 * rk(rmlp_oof)
blend_te = 0.5 * rk(fusion_te) + 0.5 * rk(rmlp_te)

print(f"融合 OOF     = {roc_auc_score(y, fusion_oof):.6f}")
print(f"RealMLP OOF  = {roc_auc_score(y, rmlp_oof):.6f}")
print(f"50/50 融合 OOF = {roc_auc_score(y, blend_oof):.6f}  Δ {roc_auc_score(y, blend_oof) - roc_auc_score(y, fusion_oof):+.6f}")

sub["satisfaction"] = blend_te
sub.to_csv("submission_realmlp_5050.csv", index=False)
print("saved -> submission_realmlp_5050.csv")
