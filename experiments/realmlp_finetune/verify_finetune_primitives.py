"""verify_finetune_primitives — fit fold0 RealMLP 后, 验证微调所需的原语可用。"""
import contextlib, io, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from pytabkit import RealMLP_TD_Classifier

DATA = "data/playground-series-s6e10"; TGT = "satisfaction"; SEED = 42
train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train[TGT].to_numpy(dtype=np.int64); ys = pd.Series(y)
X = train.drop(["id", TGT], axis=1).copy(); X_test = test.drop(["id"], axis=1).copy()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
num_cols = X.select_dtypes(include="number").columns.tolist()
category_map = {}
important_combos = [("Class","Type of Travel","Gender"),("Class","Type of Travel","Age"),("Flight Distance","Departure/Arrival time convenient")]

def feature_engineering(df, fit=False):
    for col in cat_cols: df[col] = df[col].fillna("missing")
    for col in num_cols: df[col] = df[col].fillna(0.0)
    for col in num_cols:
        cn = f"{col}_cat_"
        if fit: codes, uniques = df[col].factorize(); category_map[col] = uniques
        else: codes = df[col].map({c:i for i,c in enumerate(category_map[col])}).fillna(-1).astype("int32")
        df[cn] = codes; df[cn] = df[cn].astype("category")
    for col in cat_cols + ["Leg room service"]:
        cn = f"_{col}_count"
        if fit: cm = df[col].value_counts(); category_map[cn] = cm
        else: cm = category_map[cn]
        df[cn] = df[col].map(cm).fillna(0).astype("int32")
    combo_names = []
    for cols in important_combos:
        cn = "_".join(cols) + "_"; combo_names.append(cn)
        s = df[cols[0]].astype(str)
        for c in cols[1:]: s = s + "_" + df[c].astype(str)
        if fit: codes, uniques = pd.factorize(s, sort=False); category_map[cn] = uniques
        else: codes = s.map({c:i for i,c in enumerate(category_map[cn])}).fillna(-1).astype("int32")
        df[cn] = codes; df[cn] = df[cn].astype("category")
    return df, [c for c in df.columns if c.endswith("_")], [c for c in df.columns if c.startswith("_")], combo_names

X, _, _, combo_names = feature_engineering(X, fit=True)
cat_cols = cat_cols + [c for c in X.columns if c.endswith("_")]
num_cols = num_cols + [c for c in X.columns if c.startswith("_")]
params = dict(random_state=SEED, verbosity=0, val_metric_name="1-auc_ovr", n_ens=8, n_epochs=3, batch_size=256,
    use_early_stopping=False, lr=0.053, wd=0.015, sq_mom=0.988, lr_sched="flat_anneal", wd_sched="cos_log_15",
    first_layer_lr_factor=0.25, embedding_size=5, max_one_hot_cat_size=18, hidden_sizes=[512,256,128], act="silu",
    p_drop=0.05, p_drop_sched="invsqrtp1e-3", plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151,
    plr_sigma=2.33, ls_eps=0.01, ls_eps_sched="sqrt_cos", add_front_scale=False, bias_init_mode="neg-uniform-dynamic-2",
    tfms=["one_hot","median_center","robust_scale","smooth_clip","embedding","l2_normalize"])
params["stop_epoch"] = params["n_epochs"]

folds = list(StratifiedKFold(10, shuffle=True, random_state=42).split(X, y))
ia, iv = folds[0]
Xa, Xv = X.iloc[ia].copy(), X.iloc[iv].copy()
enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=SEED)
names = [f"_{c}TE" for c in combo_names]
Xa[names] = enc.fit_transform(Xa[combo_names], ys.iloc[ia]); Xv[names] = enc.transform(Xv[combo_names])
drop = ["Class_Type of Travel_Age_", "Flight Distance_Departure/Arrival time convenient_"]
Xa, Xv = Xa.drop(drop, axis=1), Xv.drop(drop, axis=1)

t0 = time.time()
m = RealMLP_TD_Classifier(**params)
with contextlib.redirect_stdout(io.StringIO()): m.fit(Xa, ys.iloc[ia], Xv, ys.iloc[iv])
print(f"fit ({time.time()-t0:.0f}s)", flush=True)

net = m.alg_interface_.model
print(f"\nnet 类型: {type(net).__name__}")
print(f"net.model 类型: {type(net.model).__name__}")
print(f"criterion 类型: {type(net.criterion).__name__}")

# train_dl 是否还在
print(f"\ntrain_dl 是否可迭代: {net.train_dl is not None}")
batch = next(iter(net.train_dl))
print(f"batch 类型: {type(batch).__name__}")
print(f"batch keys: {list(batch.keys()) if isinstance(batch, dict) else 'N/A'}")
if isinstance(batch, dict):
    for k, v in batch.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")

# 手动 forward
net.model.eval()
with torch.no_grad():
    out = net.model(batch)
print(f"\nforward 输出 keys: {list(out.keys())}")
for k, v in out.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")

# 优化器配置来源
print(f"\ncreator 是否保留 optimizer 配置: {hasattr(net.creator, 'config')}")
try:
    print(f"  n_ens={net.config.get('n_ens')}, lr={net.config.get('lr')}, wd={net.config.get('wd')}")
except Exception as e:
    print(f"  config 读取失败: {e}")
