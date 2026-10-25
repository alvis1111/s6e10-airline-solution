"""distill_abc — 软标签蒸馏三臂(强融合教师 → RealMLP)。

教师 = route + context + TabM(已在 fold 1-9 嵌套 CV 生成 OOF), rank 融合 + 概率校准。
三臂(同一训练器、同预算、只换监督目标):
  A: 硬标签 y
  B: 0.8·onehot(y) + 0.2·教师概率(校准后)
  C: 0.8·onehot(y) + 0.2·正例率(均匀标签平滑对照)
"""
import contextlib, io
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.isotonic import IsotonicRegression
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

outer = list(StratifiedKFold(10, shuffle=True, random_state=42).split(X, y))
ia, iv = outer[1]
Xa, Xv = X.iloc[ia].copy(), X.iloc[iv].copy()
enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=SEED)
names = [f"_{c}TE" for c in combo_names]
Xa[names] = enc.fit_transform(Xa[combo_names], ys.iloc[ia]); Xv[names] = enc.transform(Xv[combo_names])
drop = ["Class_Type of Travel_Age_", "Flight Distance_Departure/Arrival time convenient_"]
Xa, Xv = Xa.drop(drop, axis=1), Xv.drop(drop, axis=1)
y_ia, y_iv = y[ia], y[iv]

# ---- 教师: rank 融合 + 校准 ----
rte = np.load("cache/teacher_route_oof.npy")[ia]
ctx = np.load("cache/teacher_context_oof.npy")[ia]
tab = np.load("cache/teacher_tabm_oof.npy")[ia]
rk = lambda p: rankdata(p) / len(p)
fusion_rank = 0.3*rk(rte) + 0.3*rk(ctx) + 0.4*rk(tab)
iso = IsotonicRegression(out_of_bounds='clip')
teacher_prob = iso.fit_transform(fusion_rank, y_ia)   # 训练范围内校准
print(f"教师: route={roc_auc_score(y_ia, rte):.5f}  context={roc_auc_score(y_ia, ctx):.5f}  tabm={roc_auc_score(y_ia, tab):.5f}")
print(f"rank融合 OOF AUC(ia) = {roc_auc_score(y_ia, fusion_rank):.5f}  校准后 = {roc_auc_score(y_ia, teacher_prob):.5f}")

# ---- 注入通道 ----
_orig_forward = ToSoftLabelLayer.forward_tensors
def patched_forward(self, tensors):
    if 'y' in tensors and torch.is_floating_point(tensors['y']):
        return tensors
    return _orig_forward(self, tensors)
ToSoftLabelLayer.forward_tensors = patched_forward

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
    return roc_auc_score(y_iv, m.predict_proba(Xv)[:, 1])

def soft_from(p1):
    s = np.zeros((len(ia), 2), dtype=np.float32)
    s[:, 1] = p1; s[:, 0] = 1.0 - p1
    return s

prior = float(y_ia.mean())
auc_A = run(None)
auc_B = run(soft_from(0.8*y_ia + 0.2*teacher_prob))
auc_C = run(soft_from(0.8*y_ia + 0.2*prior))

print(f"\n[A] 硬标签           AUC = {auc_A:.6f}")
print(f"[B] 80%标签+20%教师   AUC = {auc_B:.6f}  Δ(B-A)={auc_B-auc_A:+.6f}")
print(f"[C] 80%标签+20%先验   AUC = {auc_C:.6f}  Δ(C-A)={auc_C-auc_A:+.6f}")
print(f"Δ(B-C) 蒸馏 vs 平滑 = {auc_B-auc_C:+.6f}")
