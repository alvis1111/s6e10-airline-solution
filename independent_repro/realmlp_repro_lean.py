"""RealMLP 精简特征版独立复现（fold 0）——Vladimir Demidov 配置，来源 busyaprime 公开源码（注明来源）。

与之前 realmlp_repro_te.py 的区别（本次验证的假设：减少工程特征 + 双表示能否让 NN 提供融合缺少的排序）：
- 输入：raw 21 特征（数值/类别双表示）+ 5 计数 + 3 交叉类别；无航线画像/教师/aux/海量 TE。
- 训练：n_ens=8、n_epochs=3、lr=0.053、wd=0.015、flat_anneal+cos_log_15。
- dropout：p_drop=0.05、p_drop_sched='invsqrtp1e-3'。
- 模型选择：stop_epoch=3（保留最后一轮，不用评分折选轮次）。
只跑 fold 0（10 折 seed 42 与自有融合对齐），TE 严格折内，保存验证行 ID + 预测，
并做「90% 原融合 + 10% RealMLP」rank 融合初筛（原融合固定 0.3·route_aux + 0.3·context + 0.4·tabm）。
"""
import contextlib
import io
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from pytabkit import RealMLP_TD_Classifier

DATA = "data/playground-series-s6e10"
TGT = "satisfaction"
SEED = 42

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train[TGT].to_numpy(dtype=np.int64)
ys = pd.Series(y)

# ---- 特征工程（busyaprime 原样：fit 于全量 train，无监督，无标签泄漏） ----
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
    for col in num_cols:  # 数值列类别副本（双表示）
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
    for col in cat_cols + ["Leg room service"]:  # 计数编码
        count_name = f"_{col}_count"
        if fit:
            count_map = df[col].value_counts()
            category_map[count_name] = count_map
        else:
            count_map = category_map[count_name]
        df[count_name] = df[col].map(count_map).fillna(0).astype("int32")
    combo_names = []
    for cols in important_combos:  # 交叉类别
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


X, new_cat_cols, new_num_cols, combo_names = feature_engineering(X, fit=True)
X_test, _, _, _ = feature_engineering(X_test, fit=False)
cat_cols = cat_cols + new_cat_cols
num_cols = num_cols + new_num_cols
print(f"特征: 数值 {len(num_cols)} / 类别 {len(cat_cols)}，总 {X.shape[1]} 列", flush=True)

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
params["stop_epoch"] = params["n_epochs"]  # 保留最后一轮，不选最佳

import sys
FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
folds = list(StratifiedKFold(10, shuffle=True, random_state=42).split(X, y))
ia, iv = folds[FOLD]
print(f"fold {FOLD}: train {len(ia)} / val {len(iv)}", flush=True)

# 折内 TE（只在 3 个交叉类别上）
Xa, Xv = X.iloc[ia].copy(), X.iloc[iv].copy()
enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=SEED + FOLD)
names = [f"_{c}TE" for c in combo_names]
Xa[names] = enc.fit_transform(Xa[combo_names], ys.iloc[ia])
Xv[names] = enc.transform(Xv[combo_names])
drop = ["Class_Type of Travel_Age_", "Flight Distance_Departure/Arrival time convenient_"]
Xa, Xv = Xa.drop(drop, axis=1), Xv.drop(drop, axis=1)

t0 = time.time()
m = RealMLP_TD_Classifier(**params)
with contextlib.redirect_stdout(io.StringIO()):
    m.fit(Xa, ys.iloc[ia], Xv, ys.iloc[iv])
pred_val = m.predict_proba(Xv)[:, 1]
single_auc = roc_auc_score(y[iv], pred_val)
print(f"fold 0 RealMLP 单模型 AUC = {single_auc:.6f}  ({time.time()-t0:.0f}s)", flush=True)

np.savez(f"cache/realmlp_lean_fold{FOLD}.npz", val_id=train["id"].iloc[iv].to_numpy(), pred=pred_val, y_val=y[iv])
print(f"saved -> cache/realmlp_lean_fold{FOLD}.npz", flush=True)

# ---- 融合初筛：90% 原融合 + 10% RealMLP（rank，仅在 fold 0 val 内） ----
route_aux = 0.5 * np.load("cache/oof_route_aux_l.npy") + 0.5 * np.load("cache/oof_route_aux_x.npy")
z = np.load("digit_confirm_results/refined_components.npz", allow_pickle=False)
order = {tid: i for i, tid in enumerate(train["id"].to_numpy())}
ctx = np.full(len(train), np.nan)
ctx[[order[t] for t in z["train_id"]]] = z["context_oof"]
tabm = np.full(len(train), np.nan)
tabm[[order[t] for t in z["train_id"]]] = z["route_tabm_oof"]
assert not np.isnan(ctx).any() and not np.isnan(tabm).any()

def rk(a):
    return rankdata(a) / len(a)

fusion = 0.3 * rk(route_aux) + 0.3 * rk(ctx) + 0.4 * rk(tabm)
base_auc = roc_auc_score(y[iv], fusion[iv])
new = 0.9 * rk(fusion[iv]) + 0.1 * rk(pred_val)
new_auc = roc_auc_score(y[iv], new)
print(f"\n原融合(fold0 val) AUC = {base_auc:.6f}", flush=True)
print(f"+10% RealMLP      AUC = {new_auc:.6f}  Δ = {new_auc-base_auc:+.6f}", flush=True)
