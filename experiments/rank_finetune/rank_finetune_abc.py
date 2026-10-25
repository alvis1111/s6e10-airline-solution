"""rank_finetune_abc — 基础网络排序目标微调 A/B/C 三臂(单折 fold 0)。

A: 冻结原模型(基线, 检查微调是否破坏性能)
B: 继续交叉熵训练(控制额外训练预算)
C: 用成对 logistic 排序损失微调(检查改变目标是否带来不同排序)

三臂从同一保存的 state_dict 出发; B/C 同训练行、同批次安排、同步数、同优化器、同随机状态。
C 损失 = 训练批次内正负样本 logit 差的 softplus 均值(逐成员计算后平均, 冻结该选择)。
"""
import contextlib, io, time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from pytabkit import RealMLP_TD_Classifier

DATA = "data/playground-series-s6e10"; TGT = "satisfaction"; SEED = 42
FINETUNE_LR = 0.005        # 固定小学习率(初筛不搜索)
FINETUNE_STEPS = 500       # 固定小预算(≈0.2 epoch)
BATCH = 256
N_ENS = 8

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
params = dict(random_state=SEED, verbosity=0, val_metric_name="1-auc_ovr", n_ens=N_ENS, n_epochs=3, batch_size=256,
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
y_tr, y_va = y[ia], y[iv]

# ---- 基础训练 + 保存 state_dict ----
t0 = time.time()
m = RealMLP_TD_Classifier(**params)
with contextlib.redirect_stdout(io.StringIO()): m.fit(Xa, ys.iloc[ia], Xv, ys.iloc[iv])
print(f"基础训练完成 ({time.time()-t0:.0f}s)", flush=True)

net = m.alg_interface_.model
base_sd = {k: v.detach().clone() for k, v in net.model.state_dict().items()}
torch.save(base_sd, "cache/realmlp_fold0_base.pt")
conv = m.x_converter_   # 复用已拟合的转换器(不能重新 fit)

# ---- 预处理一次(train + val) ----
def preprocess(df):
    ds = conv.transform(df)
    dsx = net.creator.static_model.forward_ds(ds)
    for k in list(dsx.tensors.keys()):
        t = dsx.tensors[k]
        dsx.tensors[k] = t.unsqueeze(0).expand(N_ENS, *([-1] * t.dim()))
    return dsx.tensors

tr_tensors = preprocess(Xa)     # {x_cont:[8,n,25], x_cat:[8,n,4], x_one_hot:[8,n,95]}
va_tensors = preprocess(Xv)
print(f"预处理完成: train 张量 {tr_tensors['x_cont'].shape}, val {va_tensors['x_cont'].shape}", flush=True)

# ---- 预测(softmax per member 再平均概率) ----
def predict_val():
    net.model.eval()
    with torch.no_grad():
        out = net.model(va_tensors)
    logits = out["x_cont"]                      # [8, n, 2]
    probs = torch.softmax(logits, dim=-1)[..., 1]  # [8, n]
    return probs.mean(dim=0).cpu().numpy()      # [n]

# ---- 损失 ----
ce = torch.nn.CrossEntropyLoss()
def pairwise_auc_loss(logits, yb):
    # logits [8, B, 2], yb [B] 0/1; 逐成员 pairwise, 平均
    l = logits[..., 1] - logits[..., 0]          # [8, B]
    pos = l[:, yb == 1]; neg = l[:, yb == 0]
    if pos.shape[1] == 0 or neg.shape[1] == 0:
        return logits.sum() * 0.0
    diff = pos[:, :, None] - neg[:, None, :]     # [8, np, nn]
    return torch.nn.functional.softplus(-diff).mean()

# ---- 微调循环 ----
rng = np.random.default_rng(SEED)
order = rng.permutation(len(ia))                # 共享的批次顺序
def fine_tune(loss_fn):
    net.model.load_state_dict(base_sd)          # 从同一 base 状态出发
    net.model.train()
    opt = torch.optim.AdamW(net.model.parameters(), lr=FINETUNE_LR, weight_decay=0.015)
    for step in range(FINETUNE_STEPS):
        idx = order[step * BATCH:(step + 1) * BATCH]
        batch = {k: v[:, idx] for k, v in tr_tensors.items()}
        out = net.model(batch)
        logits = out["x_cont"]                   # [8, B, 2]
        yb = torch.as_tensor(y_tr[idx], device=logits.device)
        loss = loss_fn(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    return predict_val()

# ---- A: 冻结 ----
net.model.load_state_dict(base_sd)
auc_A = roc_auc_score(y_va, predict_val())
print(f"\n[A] 冻结原模型     AUC = {auc_A:.6f}", flush=True)

# ---- B: 继续 CE ----
t0 = time.time()
def ce_loss(logits, yb):
    return ce(logits.reshape(-1, 2), yb.repeat(N_ENS))
pred_B = fine_tune(ce_loss)
auc_B = roc_auc_score(y_va, pred_B)
print(f"[B] 继续CE微调     AUC = {auc_B:.6f}  Δ(B-A)={auc_B-auc_A:+.6f}  ({time.time()-t0:.0f}s)", flush=True)

# ---- C: pairwise 排序微调 ----
t0 = time.time()
pred_C = fine_tune(pairwise_auc_loss)
auc_C = roc_auc_score(y_va, pred_C)
print(f"[C] pairwise微调   AUC = {auc_C:.6f}  Δ(C-A)={auc_C-auc_A:+.6f}  Δ(C-B)={auc_C-auc_B:+.6f}  ({time.time()-t0:.0f}s)", flush=True)

print(f"\n结论: A={auc_A:.6f}  B={auc_B:.6f}  C={auc_C:.6f}", flush=True)
