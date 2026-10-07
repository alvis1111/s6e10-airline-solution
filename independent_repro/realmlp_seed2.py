"""RealMLP 第二种子两折小实验：只换模型随机种子，新旧概率等权平均，50/50 融合配对比较。

保持特征、折划分(10折 seed 42)、TE(random_state=42+fold)、训练参数不变，只改
params['random_state'] 与 torch.manual_seed。fold 0/1 上跑第二种子，与 seed 42 概率等权平均，
按 50/50 配方(0.5·rank(融合)+0.5·rank(realmlp))融合，配对 DeLong 比单种子。
"""
import contextlib
import io
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from pytabkit import RealMLP_TD_Classifier
from delong import delong_ci

DATA = "data/playground-series-s6e10"
TGT = "satisfaction"
BASE_SEED = 42
NEW_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 43

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
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
            codes, uniques = df[col].factorize(); category_map[col] = uniques
        else:
            uniques = category_map[col]
            code_map = {c: i for i, c in enumerate(uniques)}
            codes = df[col].map(code_map).fillna(-1).astype("int32")
        df[cat_name] = codes
        df[cat_name] = df[cat_name].astype("category")
    for col in cat_cols + ["Leg room service"]:
        count_name = f"_{col}_count"
        if fit:
            count_map = df[col].value_counts(); category_map[count_name] = count_map
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
            codes, uniques = pd.factorize(combo_series, sort=False); category_map[combo_name] = uniques
        else:
            uniques = category_map[combo_name]
            code_map = {c: i for i, c in enumerate(uniques)}
            codes = combo_series.map(code_map).fillna(-1).astype("int32")
        df[combo_name] = codes
        df[combo_name] = df[combo_name].astype("category")
    return df, combo_names


X, combo_names = feature_engineering(X, fit=True)
X_test, _ = feature_engineering(X_test, fit=False)
print(f"特征 {X.shape[1]} 列", flush=True)


def params_for(seed):
    p = {
        "random_state": seed, "verbosity": 0, "val_metric_name": "1-auc_ovr",
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
    p["stop_epoch"] = p["n_epochs"]
    return p


folds = list(StratifiedKFold(10, shuffle=True, random_state=42).split(X, y))
drop = ["Class_Type of Travel_Age_", "Flight Distance_Departure/Arrival time convenient_"]


def run_fold(seed, fold):
    ia, iv = folds[fold]
    Xa, Xv = X.iloc[ia].copy(), X.iloc[iv].copy()
    enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=BASE_SEED + fold)  # TE 不变
    names = [f"_{c}TE" for c in combo_names]
    Xa[names] = enc.fit_transform(Xa[combo_names], ys.iloc[ia])
    Xv[names] = enc.transform(Xv[combo_names])
    Xa, Xv = Xa.drop(drop, axis=1), Xv.drop(drop, axis=1)
    torch.manual_seed(seed)
    m = RealMLP_TD_Classifier(**params_for(seed))
    with contextlib.redirect_stdout(io.StringIO()):
        m.fit(Xa, ys.iloc[ia], Xv, ys.iloc[iv])
    pred = m.predict_proba(Xv)[:, 1]
    auc = roc_auc_score(y[iv], pred)
    del m; torch.cuda.empty_cache()
    return iv, pred, auc


print(f"第二种子 = {NEW_SEED}", flush=True)
new_preds = {}
for fold in [0, 1]:
    t0 = time.time()
    iv, pred, auc = run_fold(NEW_SEED, fold)
    new_preds[fold] = (iv, pred)
    np.savez(f"cache/realmlp_lean_s{NEW_SEED}_fold{fold}.npz", val_id=train["id"].iloc[iv].to_numpy(), pred=pred, y_val=y[iv])
    print(f"seed {NEW_SEED} fold {fold}: 单模型 {auc:.6f}  ({time.time()-t0:.0f}s)", flush=True)

# ---- 50/50 融合：单种子 vs 两种子 ----
route_aux = 0.5 * np.load("cache/oof_route_aux_l.npy") + 0.5 * np.load("cache/oof_route_aux_x.npy")
z = np.load("digit_confirm_results/refined_components.npz", allow_pickle=False)
order = {tid: i for i, tid in enumerate(train["id"].to_numpy())}
ctx = np.full(len(train), np.nan); ctx[[order[t] for t in z["train_id"]]] = z["context_oof"]
tabm = np.full(len(train), np.nan); tabm[[order[t] for t in z["train_id"]]] = z["route_tabm_oof"]
rk = lambda a: rankdata(a) / len(a)
fusion = 0.3 * rk(route_aux) + 0.3 * rk(ctx) + 0.4 * rk(tabm)

ys_, single, two = [], [], []
for fold in [0, 1]:
    d42 = np.load(f"cache/realmlp_lean_fold{fold}.npz")
    iv, p42 = d42["val_id"], d42["pred"]
    iv2, p43 = new_preds[fold]
    assert np.array_equal(iv, iv2)
    yv = d42["y_val"]
    f = fusion[iv]
    single.append(0.5 * rk(f) + 0.5 * rk(p42))
    two.append(0.5 * rk(f) + 0.5 * rk(0.5 * (p42 + p43)))
    ys_.append(yv)

y_all = np.concatenate(ys_); s_all = np.concatenate(single); t_all = np.concatenate(two)
print(f"\n合并两折 {len(y_all)} 行", flush=True)
print(f"单种子 50/50 AUC {roc_auc_score(y_all, s_all):.6f}  两种子 50/50 AUC {roc_auc_score(y_all, t_all):.6f}", flush=True)
_, _, delta, lo, hi, p = delong_ci(y_all, s_all, t_all)
print(f"配对 DeLong: Δ={delta:+.7f}  95%CI=[{lo:+.7f}, {hi:+.7f}]  p={p:.4f}", flush=True)
