"""RealMLP 精简版完整 10 折训练 + 原融合 90/10 提交（来源 busyaprime/Demidov 配置注明）。

= realmlp_repro_lean.py 的 10 折版：逐折 checkpoint（cache/partial/realmlp_full_f*.npz），
  折内 TE、stop_epoch=3，测试预测跨折平均；完成后和原融合（0.3·route_aux+0.3·context+0.4·tabm）
  做 90/10 rank 融合生成提交。
"""
import contextlib
import io
import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from pytabkit import RealMLP_TD_Classifier

DATA = "data/playground-series-s6e10"
TGT = "satisfaction"
SEED = 42
NFOLD = 10

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
sub = pd.read_csv(f"{DATA}/sample_submission.csv")
y = train[TGT].to_numpy(dtype=np.int64)
ys = pd.Series(y)

X = train.drop(["id", TGT], axis=1).copy()
X_test = test.drop(["id"], axis=1).copy()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
num_cols = X.select_dtypes(include="number").columns.tolist()
category_map = {}
important_combos = [
    ("Class", "Type of Travel", "Gender"),
    ("Class", "Type of Travel", "Age"),
    ("Flight Distance", "Departure/Arrival time convenient"),
]


def feature_engineering(df, fit=False):
    for col in cat_cols:
        df[col] = df[col].fillna("missing")
    for col in num_cols:
        df[col] = df[col].fillna(0.0)
    for col in num_cols:
        cat_name = f"{col}_cat_"
        if fit:
            codes, uniques = df[col].factorize()
            category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = df[col].map(code_map).fillna(-1).astype("int32")
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype("category")
    for col in cat_cols + ["Leg room service"]:
        count_name = f"_{col}_count"
        if fit:
            count_map = df[col].value_counts()
            category_map[count_name] = count_map
        else:
            count_map = category_map[count_name]
        df[count_name] = df[col].map(count_map).fillna(0).astype("int32")
    combo_names = []
    for cols in important_combos:
        combo_name = "_".join(cols) + "_"
        combo_names.append(combo_name)
        combo_series = df[cols[0]].astype(str)
        for col in cols[1:]:
            combo_series = combo_series + "_" + df[col].astype(str)
        if fit:
            codes, uniques = pd.factorize(combo_series, sort=False)
            category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {cat: i for i, cat in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype("int32")
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype("category")
    new_cat_cols = [col for col in df.columns if col.endswith("_")]
    new_num_cols = [col for col in df.columns if col.startswith("_")]
    return df, new_cat_cols, new_num_cols, combo_names


X, _, _, combo_names = feature_engineering(X, fit=True)
X_test, _, _, _ = feature_engineering(X_test, fit=False)
print(f"特征 {X.shape[1]} 列", flush=True)

params = {
    "random_state": SEED, "verbosity": 0, "val_metric_name": "1-auc_ovr",
    "n_ens": 8, "n_epochs": 3, "batch_size": 256,
    "use_early_stopping": False,
    "early_stopping_additive_patience": 10, "early_stopping_multiplicative_patience": 1,
    "lr": 0.053, "wd": 0.015, "sq_mom": 0.988,
    "lr_sched": "flat_anneal", "wd_sched": "cos_log_15", "first_layer_lr_factor": 0.25,
    "embedding_size": 5, "max_one_hot_cat_size": 18,
    "hidden_sizes": [512, 256, 128], "act": "silu",
    "p_drop": 0.05, "p_drop_sched": "invsqrtp1e-3",
    "plr_hidden_1": 16, "plr_hidden_2": 8, "plr_act_name": "gelu", "plr_lr_factor": 0.1151, "plr_sigma": 2.33,
    "ls_eps": 0.01, "ls_eps_sched": "sqrt_cos",
    "add_front_scale": False, "bias_init_mode": "neg-uniform-dynamic-2",
    "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
}
params["stop_epoch"] = params["n_epochs"]

os.makedirs("cache/partial", exist_ok=True)
folds = list(StratifiedKFold(NFOLD, shuffle=True, random_state=42).split(X, y))
oof = np.zeros(len(train), dtype=np.float64)
test_pred = np.zeros(len(test), dtype=np.float64)
drop = ["Class_Type of Travel_Age_", "Flight Distance_Departure/Arrival time convenient_"]
t_start = time.time()

for fold, (ia, iv) in enumerate(folds):
    part = f"cache/partial/realmlp_full_f{fold}.npz"
    if os.path.exists(part):
        d = np.load(part)
        oof[iv] = d["oof"]; test_pred += d["test"] / NFOLD
        print(f"fold {fold} restored  auc {roc_auc_score(y[iv], oof[iv]):.6f}", flush=True)
        continue
    t0 = time.time()
    Xa, Xv = X.iloc[ia].copy(), X.iloc[iv].copy()
    Xtk = X_test.copy()
    enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=SEED + fold)
    names = [f"_{c}TE" for c in combo_names]
    Xa[names] = enc.fit_transform(Xa[combo_names], ys.iloc[ia])
    Xv[names] = enc.transform(Xv[combo_names]); Xtk[names] = enc.transform(Xtk[combo_names])
    Xa, Xv, Xtk = Xa.drop(drop, axis=1), Xv.drop(drop, axis=1), Xtk.drop(drop, axis=1)
    m = RealMLP_TD_Classifier(**params)
    with contextlib.redirect_stdout(io.StringIO()):
        m.fit(Xa, ys.iloc[ia], Xv, ys.iloc[iv])
    oof[iv] = m.predict_proba(Xv)[:, 1]
    pt = m.predict_proba(Xtk)[:, 1]
    test_pred += pt / NFOLD
    np.savez(part, oof=oof[iv], test=pt)
    print(f"fold {fold}: auc {roc_auc_score(y[iv], oof[iv]):.6f}  ({time.time()-t0:.0f}s)", flush=True)
    del m; torch.cuda.empty_cache()

print(f"\nRealMLP 10折 OOF AUC = {roc_auc_score(y, oof):.6f}  (总 {time.time()-t_start:.0f}s)", flush=True)
np.save("cache/oof_realmlp_lean.npy", oof)
np.save("cache/pred_realmlp_lean.npy", test_pred)

# ---- 原融合 90/10 rank 融合提交 ----
route_aux = 0.5 * np.load("cache/oof_route_aux_l.npy") + 0.5 * np.load("cache/oof_route_aux_x.npy")
route_aux_te = 0.5 * np.load("cache/pred_route_aux_l.npy") + 0.5 * np.load("cache/pred_route_aux_x.npy")
z = np.load("digit_confirm_results/refined_components.npz", allow_pickle=False)
tr_ord = {tid: i for i, tid in enumerate(train["id"].to_numpy())}
te_ord = {tid: i for i, tid in enumerate(test["id"].to_numpy())}
ctx = np.full(len(train), np.nan); ctx[[tr_ord[t] for t in z["train_id"]]] = z["context_oof"]
tabm = np.full(len(train), np.nan); tabm[[tr_ord[t] for t in z["train_id"]]] = z["route_tabm_oof"]
ctx_z = np.load("C:/Users/GEM07/OneDrive/Desktop/s6e10_outer10_blend/previous_context_ensemble.npz", allow_pickle=False)
ctx_te = np.full(len(test), np.nan)
ctx_te[[te_ord[t] for t in ctx_z["test_id"]]] = ctx_z["pred"]
tabm_te = np.load("cache/pred_route_tabm.npy")
assert not np.isnan(ctx).any() and not np.isnan(tabm).any() and not np.isnan(ctx_te).any()

rk = lambda a: rankdata(a) / len(a)
fusion_oof = 0.3 * rk(route_aux) + 0.3 * rk(ctx) + 0.4 * rk(tabm)
fusion_te = 0.3 * rk(route_aux_te) + 0.3 * rk(ctx_te) + 0.4 * rk(tabm_te)
print(f"原融合 OOF = {roc_auc_score(y, fusion_oof):.6f}", flush=True)

blend_oof = 0.9 * rk(fusion_oof) + 0.1 * rk(oof)
blend_te = 0.9 * rk(fusion_te) + 0.1 * rk(test_pred)
print(f"融合+10%RealMLP OOF = {roc_auc_score(y, blend_oof):.6f}  Δ {roc_auc_score(y, blend_oof)-roc_auc_score(y, fusion_oof):+.6f}", flush=True)

sub["satisfaction"] = blend_te
sub.to_csv("submission_realmlp_lean.csv", index=False)
print("saved -> submission_realmlp_lean.csv", flush=True)
