"""用 blend_ours.py 网格搜出的 5 路权重生成提交 CSV。"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

train = pd.read_csv("data/playground-series-s6e10/train.csv")
test = pd.read_csv("data/playground-series-s6e10/test.csv")
sub = pd.read_csv("data/playground-series-s6e10/sample_submission.csv")
y = train["satisfaction"].to_numpy(np.int64)
rk = lambda a: rankdata(a) / len(a)

# OOF
route_l = np.load("cache/oof_route_aux_l.npy"); route_x = np.load("cache/oof_route_aux_x.npy")
rmlp = np.load("cache/oof_realmlp_lean.npy")
z = np.load("digit_confirm_results/refined_components.npz", allow_pickle=False)
tr_ord = {tid: i for i, tid in enumerate(train["id"].to_numpy())}
ctx = np.full(len(train), np.nan); ctx[[tr_ord[t] for t in z["train_id"]]] = z["context_oof"]
tabm = np.full(len(train), np.nan); tabm[[tr_ord[t] for t in z["train_id"]]] = z["route_tabm_oof"]
assert not np.isnan(ctx).any() and not np.isnan(tabm).any()

# test
route_l_te = np.load("cache/pred_route_aux_l.npy"); route_x_te = np.load("cache/pred_route_aux_x.npy")
rmlp_te = np.load("cache/pred_realmlp_lean.npy")
tabm_te = np.load("cache/pred_route_tabm.npy")
cz = np.load("C:/Users/GEM07/OneDrive/Desktop/s6e10_outer10_blend/previous_context_ensemble.npz", allow_pickle=False)
te_ord = {tid: i for i, tid in enumerate(test["id"].to_numpy())}
ctx_te = np.full(len(test), np.nan); ctx_te[[te_ord[t] for t in cz["test_id"]]] = cz["pred"]
assert not np.isnan(ctx_te).any()

# 网格权重（route_lgb, route_xgb, context, tabm, realmlp）
W = np.array([0.2083, 0.2083, 0.1667, 0.0417, 0.3750])
assert abs(W.sum() - 1.0) < 1e-6

OOF = [route_l, route_x, ctx, tabm, rmlp]
TE = [route_l_te, route_x_te, ctx_te, tabm_te, rmlp_te]

oof_blend = sum(w * rk(a) for w, a in zip(W, OOF))
te_blend = sum(w * rk(a) for w, a in zip(W, TE))
print(f"5 路 OOF = {roc_auc_score(y, oof_blend):.6f}")

sub["satisfaction"] = te_blend
sub.to_csv("submission_blend_ours_5leg.csv", index=False)
print("saved -> submission_blend_ours_5leg.csv")
