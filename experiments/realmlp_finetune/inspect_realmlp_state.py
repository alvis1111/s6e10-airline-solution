"""inspect_realmlp_state — fit fold0 的 RealMLP, 递归定位内部 torch 模型并验证 state_dict 可提取。
仅探查 + 验证, 不改动训练配方。
"""
import contextlib, io, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from pytabkit import RealMLP_TD_Classifier

DATA = "data/playground-series-s6e10"
TGT = "satisfaction"
SEED = 42

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

X, new_cat_cols, new_num_cols, combo_names = feature_engineering(X, fit=True)
cat_cols = cat_cols + new_cat_cols
num_cols = num_cols + new_num_cols

params = {
    "random_state": SEED, "verbosity": 0, "val_metric_name": "1-auc_ovr",
    "n_ens": 8, "n_epochs": 3, "batch_size": 256,
    "use_early_stopping": False,
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

folds = list(StratifiedKFold(10, shuffle=True, random_state=42).split(X, y))
ia, iv = folds[0]
Xa, Xv = X.iloc[ia].copy(), X.iloc[iv].copy()
enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=SEED)
names = [f"_{c}TE" for c in combo_names]
Xa[names] = enc.fit_transform(Xa[combo_names], ys.iloc[ia])
Xv[names] = enc.transform(Xv[combo_names])
drop = ["Class_Type of Travel_Age_", "Flight Distance_Departure/Arrival time convenient_"]
Xa, Xv = Xa.drop(drop, axis=1), Xv.drop(drop, axis=1)

t0 = time.time()
m = RealMLP_TD_Classifier(**params)
with contextlib.redirect_stdout(io.StringIO()):
    m.fit(Xa, ys.iloc[ia], Xv, ys.iloc[iv])
print(f"fit 完成 ({time.time()-t0:.0f}s)", flush=True)
pred = m.predict_proba(Xv)[:, 1]

# 递归定位 nn.Module
def find_modules(obj, prefix="", depth=0, out=None, seen=None):
    if out is None: out = []
    if seen is None: seen = set()
    if depth > 5 or id(obj) in seen: return out
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        n_params = sum(p.numel() for p in obj.parameters())
        out.append((prefix or type(obj).__name__, type(obj).__name__, n_params))
    for attr in dir(obj):
        if attr.startswith("__"): continue
        try:
            v = getattr(obj, attr)
        except Exception:
            continue
        if isinstance(v, (torch.nn.Module, list, tuple)) or (isinstance(v, object) and hasattr(v, "__dict__")):
            find_modules(v, f"{prefix}.{attr}", depth+1, out, seen)
    return out

mods = find_modules(m)
print(f"\n=== 找到 {len(mods)} 个 nn.Module ===")
for prefix, typ, nparams in mods:
    print(f"  {prefix}  ({typ}, {nparams:,} params)")

# ---- state_dict 保存/加载 roundtrip 验证 ----
net = m.alg_interface_.model
sd = net.state_dict()
print(f"\n=== state_dict: {len(sd)} 个张量, 共 {sum(v.numel() for v in sd.values()):,} 参数 ===")
torch.save(sd, "cache/realmlp_fold0_state.pt")
sd2 = torch.load("cache/realmlp_fold0_state.pt", weights_only=False)
# 逐 key 校验一致
keys_match = list(sd.keys()) == list(sd2.keys())
vals_close = all(torch.equal(sd[k], sd2[k]) for k in sd)
print(f"keys 一致: {keys_match}, 值逐张量一致: {vals_close}")

# 加载回原模型, 验证预测不变
net.load_state_dict(sd2)
pred2 = m.predict_proba(Xv)[:, 1]
max_diff = float(np.max(np.abs(pred - pred2)))
print(f"预测最大差异: {max_diff:.3e}  (应为 0)")
print(f"roundtrip 验证: {'通过 ✅' if max_diff < 1e-6 else '失败 ❌'}")

