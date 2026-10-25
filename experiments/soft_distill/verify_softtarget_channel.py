"""verify_softtarget_channel — 验证软目标注入通道(ToSoftLabelLayer 透传 float y)。"""
import contextlib, io
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from pytabkit import RealMLP_TD_Classifier
from pytabkit.models.nn_models.nn import ToSoftLabelLayer
from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface

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
params = dict(random_state=SEED, verbosity=0, val_metric_name="1-auc_ovr", n_ens=8, n_epochs=3, batch_size=256,
    use_early_stopping=False, lr=0.053, wd=0.015, sq_mom=0.988, lr_sched="flat_anneal", wd_sched="cos_log_15",
    first_layer_lr_factor=0.25, embedding_size=5, max_one_hot_cat_size=18, hidden_sizes=[512,256,128], act="silu",
    p_drop=0.05, p_drop_sched="invsqrtp1e-3", plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151,
    plr_sigma=2.33, ls_eps=0.0, ls_eps_sched="sqrt_cos", add_front_scale=False, bias_init_mode="neg-uniform-dynamic-2",
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

# ---- 关键 patch: ToSoftLabelLayer 对 float y 透传 ----
_orig_forward = ToSoftLabelLayer.forward_tensors
def patched_forward(self, tensors):
    if 'y' in tensors and torch.is_floating_point(tensors['y']):
        return tensors  # 已是软目标(one-hot float), 直接透传
    return _orig_forward(self, tensors)
ToSoftLabelLayer.forward_tensors = patched_forward

# ---- 注入软目标(在原始 ds 层, 替换 y 为 float 软目标) ----
_SOFT_TARGET = None
class SoftTargetNNAlgInterface(NNAlgInterface):
    def fit(self, ds=None, *args, **kwargs):
        if _SOFT_TARGET is not None and ds is not None:
            ds.tensors['y'] = torch.as_tensor(_SOFT_TARGET, dtype=torch.float32)
        return super().fit(ds=ds, *args, **kwargs)

class SoftTargetRealMLP(RealMLP_TD_Classifier):
    def _create_alg_interface(self, n_cv):
        return SoftTargetNNAlgInterface(**self.get_config())

def run(soft_target):
    global _SOFT_TARGET
    _SOFT_TARGET = soft_target
    m = SoftTargetRealMLP(**params)
    with contextlib.redirect_stdout(io.StringIO()): m.fit(Xa, ys.iloc[ia], Xv, ys.iloc[iv])
    return roc_auc_score(y[iv], m.predict_proba(Xv)[:, 1])

auc_A = run(None)
onehot = np.zeros((len(ia), 2), dtype=np.float32)
onehot[np.arange(len(ia)), ys.iloc[ia].to_numpy()] = 1.0
auc_Aprime = run(onehot)

print(f"\n[A] 硬标签(无注入)     AUC = {auc_A:.6f}")
print(f"[A'] 硬 one-hot 软目标   AUC = {auc_Aprime:.6f}")
print(f"Δ(A' - A) = {auc_Aprime-auc_A:+.6f}")
print(f"注入通道: {'通过 ✅' if abs(auc_Aprime-auc_A) < 1e-4 else '未复现 ❌'}")
