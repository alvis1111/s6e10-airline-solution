"""verify_forward_reconstruction — 重建 DataFrame -> DictDataset -> static_model -> forward 链路, 与 predict_proba 对齐。"""
import contextlib, io, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from pytabkit import RealMLP_TD_Classifier
from pytabkit.models.data.conversion import ToDictDatasetConverter

DATA = "data/playground-series-s6e10"; TGT = "satisfaction"; SEED = 42
train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv")
y = train[TGT].to_numpy(dtype=np.int64); ys = pd.Series(y)
X = train.drop(["id", TGT], axis=1).copy(); X_test = test.drop(["id"], axis=1).copy()
cat_cols = X.select_dtypes(exclude="number").columns.tolist(); num_cols = X.select_dtypes(include="number").columns.tolist()
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
# 重建 forward: DataFrame -> DictDataset -> static_model -> net.model
conv = ToDictDatasetConverter()
ds = conv.fit_transform(Xa)          # DictDataset{x_cont, x_cat}
print(f"\nDictDataset: x_cont={ds.tensors['x_cont'].shape}, x_cat={ds.tensors['x_cat'].shape}")

ds_x = net.creator.static_model.forward_ds(ds)
print(f"static_model 后 keys: {list(ds_x.tensors.keys())}")
for k, v in ds_x.tensors.items():
    print(f"  {k}: {v.shape}")

# n_ens 复制(模型内部 8 成员在 leading 维)
n_ens = net.config.get('n_ens', 1)
for k in list(ds_x.tensors.keys()):
    t = ds_x.tensors[k]
    ds_x.tensors[k] = t.unsqueeze(0).expand(n_ens, *([-1] * t.dim()))
print(f"复制 n_ens={n_ens} 后:")
for k, v in ds_x.tensors.items():
    print(f"  {k}: {v.shape}")

net.model.eval()
with torch.no_grad():
    out = net.model(ds_x)
print(f"\nforward 输出类型: {type(out).__name__}, keys: {list(out.tensors.keys())}")
logits = out.tensors["x_cont"]
print(f"logits shape: {logits.shape}")

# 与 predict_proba 对齐(train 行)
pred = m.predict_proba(Xa)[:, 1]
logit_pred = np.log(np.clip(pred, 1e-6, 1-1e-6) / (1-np.clip(pred, 1e-6, 1-1e-6)))
logits_np = logits.detach().cpu().numpy().mean(axis=0)   # [n, 2] 两分类 logits
single_logit = logits_np[:, 1] - logits_np[:, 0]          # [n] 单 logit
diff = np.max(np.abs(single_logit - logit_pred))
print(f"\n重建 logit 与 predict_proba 最大差异: {diff:.4e}")
print(f"forward 重建: {'通过 ✅' if diff < 0.01 else '需检查'}")
# 顺带确认: 用重建 logit 算 val AUC(应接近模型自身)
from sklearn.metrics import roc_auc_score
conv_v = ToDictDatasetConverter(); ds_v = conv_v.fit_transform(Xv)
ds_vx = net.creator.static_model.forward_ds(ds_v)
for k in list(ds_vx.tensors.keys()):
    t = ds_vx.tensors[k]; ds_vx.tensors[k] = t.unsqueeze(0).expand(n_ens, *([-1] * t.dim()))
with torch.no_grad():
    out_v = net.model(ds_vx)
lgt_v = out_v.tensors["x_cont"].detach().cpu().numpy().mean(axis=0)
sl_v = lgt_v[:, 1] - lgt_v[:, 0]
print(f"重建 val AUC = {roc_auc_score(y[iv], sl_v):.6f}  (模型自身 = {roc_auc_score(y[iv], m.predict_proba(Xv)[:,1]):.6f})")
