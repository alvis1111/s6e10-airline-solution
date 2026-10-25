"""transfer_adapter — 冻结预处理+网络init的局部适配器 + 零步加载检查。

模板(n_epochs=0 在 ia)捕获 frozen converter/static_model/网络init。
monkey-patch: converter.fit_transform 用冻结 transform; create_model 复用冻结 static_model + 深拷贝 init 网络(或 warm 状态)。
零步检查: 相同样本/批/精度下, 保存前后预测一致。
"""
import contextlib, io, copy
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from pytabkit import RealMLP_TD_Classifier
from pytabkit.models.data.conversion import ToDictDatasetConverter
from pytabkit.models.training.nn_creator import NNCreator

SEED = 42
y = pd.read_csv("data/playground-series-s6e10/train.csv")["satisfaction"].to_numpy(np.int64)
d = np.load("cache/transfer_dataprep.npz"); ia, iv = d["ia"], d["iv"]
Xa = pd.read_pickle("cache/transfer_Xa.pkl"); Xv = pd.read_pickle("cache/transfer_Xv.pkl")
O = pd.read_pickle("cache/transfer_O.pkl"); oy = pd.read_pickle("cache/transfer_oy.pkl")["satisfaction"].to_numpy(np.int64)

params = dict(random_state=SEED, verbosity=0, val_metric_name="1-auc_ovr", n_ens=8, n_epochs=3, batch_size=256,
    use_early_stopping=False, lr=0.053, wd=0.015, sq_mom=0.988, lr_sched="flat_anneal", wd_sched="cos_log_15",
    first_layer_lr_factor=0.25, embedding_size=5, max_one_hot_cat_size=18, hidden_sizes=[512,256,128], act="silu",
    p_drop=0.05, p_drop_sched="invsqrtp1e-3", plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151,
    plr_sigma=2.33, ls_eps=0.0, ls_eps_sched="sqrt_cos", add_front_scale=False, bias_init_mode="neg-uniform-dynamic-2",
    tfms=["one_hot","median_center","robust_scale","smooth_clip","embedding","l2_normalize"])
params["stop_epoch"] = params["n_epochs"]

# ---- 建模板(n_epochs=0, 捕获冻结组件) ----
m_template = RealMLP_TD_Classifier(**dict(params, n_epochs=0))
with contextlib.redirect_stdout(io.StringIO()):
    m_template.fit(Xa, y[ia], Xv, y[iv])
FROZEN = {
    'converter': m_template.x_converter_,
    'static': m_template.alg_interface_.model.creator.static_model,
    'net': copy.deepcopy(m_template.alg_interface_.model.model),  # 未训练 init 网络
}
pred_template = m_template.predict_proba(Xv)[:, 1]
print(f"模板捕获: 网络参数量={sum(p.numel() for p in FROZEN['net'].parameters()):,}", flush=True)

# ---- monkey-patch ----
_WARM_SD = None
_orig_ft = ToDictDatasetConverter.fit_transform
def patched_ft(self, x):
    # 复制冻结 converter 的全部状态, 使后续 self.transform 可用
    for k, v in FROZEN['converter'].__dict__.items():
        setattr(self, k, v)
    return FROZEN['converter'].transform(x)   # 冻结 transform
ToDictDatasetConverter.fit_transform = patched_ft

_orig_cm = NNCreator.create_model
def patched_cm(self, ds, idxs_list):
    self.static_model = FROZEN['static']       # 冻结 static_model(含 data_tfm)
    net = copy.deepcopy(FROZEN['net'])         # 深拷贝 init 网络
    if _WARM_SD is not None:
        net.load_state_dict(_WARM_SD)          # 微调时加载预训练状态
    return net
NNCreator.create_model = patched_cm

def fit_eval(Xtr, ytr, Xva, yva, n_epochs, warm_sd=None):
    global _WARM_SD; _WARM_SD = warm_sd
    p = dict(params, n_epochs=n_epochs); p["stop_epoch"] = n_epochs
    m = RealMLP_TD_Classifier(**p)
    with contextlib.redirect_stdout(io.StringIO()):
        m.fit(Xtr, ytr, Xva, yva)
    return m

# ---- 零步加载检查: 相同样本/批/精度下, 保存前后预测一致 ----
m_zero = fit_eval(Xa, y[ia], Xv, y[iv], n_epochs=0)   # 冻结后 n_epochs=0, 未训练
pred_zero = m_zero.predict_proba(Xv)[:, 1]
max_diff = float(np.max(np.abs(pred_zero - pred_template)))
print(f"零步加载检查: 预测最大差 = {max_diff:.3e}  {'通过 ✅' if max_diff < 1e-5 else '未通过 ❌'}", flush=True)

# ---- B/C 训练 ----
def get_sd(m):
    return {k: v.clone() for k, v in m.alg_interface_.model.model.state_dict().items()}
def eval_iv(m):
    return roc_auc_score(y[iv], m.predict_proba(Xv)[:, 1])

# B: 预训练 O(1 epoch, 冻结 init) → 微调 ia(3 epoch, 预训练状态)
m_B1 = fit_eval(O, oy, Xv, y[iv], n_epochs=1, warm_sd=None)
sd_B = get_sd(m_B1)
m_B = fit_eval(Xa, y[ia], Xv, y[iv], n_epochs=3, warm_sd=sd_B)
auc_B = eval_iv(m_B)
print(f"[B] 预训练O→微调ia: AUC={auc_B:.6f}", flush=True)

# C: 预热 ia 等量子集(1 epoch) → 微调 ia(3 epoch)
rng = np.random.default_rng(SEED)
ia_sub = rng.choice(len(Xa), size=len(O), replace=False)
m_C1 = fit_eval(Xa.iloc[ia_sub], y[ia][ia_sub], Xv, y[iv], n_epochs=1, warm_sd=None)
sd_C = get_sd(m_C1)
m_C = fit_eval(Xa, y[ia], Xv, y[iv], n_epochs=3, warm_sd=sd_C)
auc_C = eval_iv(m_C)
print(f"[C] 预热ia→微调ia: AUC={auc_C:.6f}", flush=True)

AUC_A = 0.961491
print(f"\n=== 三臂结果(fold 0, 同验证行) ===", flush=True)
print(f"A 从头训练      = {AUC_A:.6f}", flush=True)
print(f"B 预训练O→微调  = {auc_B:.6f}  Δ(B-A)={auc_B-AUC_A:+.6f}", flush=True)
print(f"C 预热ia→微调   = {auc_C:.6f}  Δ(C-A)={auc_C-AUC_A:+.6f}", flush=True)
print(f"Δ(B-C) = {auc_B-auc_C:+.6f}", flush=True)
