"""S6E10 — 复现 PyTabKit RealMLP（goodpjw2008 配方），检查对现有融合的互补性。

特征：raw 21 列(4类作字符串) + 航线画像(fdm_* 按精确距离均值) + 值计数(cnt_*) +
orig_proba/orig_logit + aux_ev(13)。全部 fold-invariant（无标签），5 折 seed 42 对齐。
用 goodpjw2008 的 REALMLP 配方（flat_anneal、6 epochs、n_ens=8）。
"""
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DATA = "data/playground-series-s6e10"
REAL = r"C:\Users\GEM07\.cache\kagglehub\datasets\teejmahal20\airline-passenger-satisfaction\versions\1"

RATING_COLS = ["Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
    "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness"]
CAT_COLS_RAW = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
FEATS = NUM_COLS + RATING_COLS + CAT_COLS_RAW
AUX_COLS = [f"aux_ev_{c}" for c in RATING_COLS]

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train["satisfaction"].to_numpy(dtype=np.int64)
NTR = len(train)

# ---- orig 模型（原始数据）----
orig = pd.concat([pd.read_csv(f"{REAL}/train.csv"), pd.read_csv(f"{REAL}/test.csv")], ignore_index=True)
orig["satisfaction"] = (orig["satisfaction"] == "satisfied").astype(int)
orig = orig.drop(columns=["Unnamed: 0", "id", "Inflight service"], errors="ignore")
def row_hashes(df): return pd.util.hash_pandas_object(df[FEATS].astype(str), index=False)
comp = pd.concat([row_hashes(train), row_hashes(test)], ignore_index=True)
oh = row_hashes(orig).isin(comp)
orig = orig[~oh].drop_duplicates().reset_index(drop=True)
oy = orig["satisfaction"].astype(int)
ovocab = {}
for c in CAT_COLS_RAW:
    cats = sorted(orig[c].dropna().unique()); ovocab[c] = cats
    orig[c] = pd.Categorical(orig[c], categories=cats)
ox = orig[FEATS].copy()
for c in FEATS:
    if c not in CAT_COLS_RAW: ox[c] = ox[c].astype("float32")
om = xgb.XGBClassifier(objective="binary:logistic", n_estimators=600, learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.5, tree_method="hist", enable_categorical=True,
    n_jobs=-1, random_state=0, verbosity=0).fit(ox, oy)
def orig_pred(df):
    Xr = df[FEATS].copy()
    for c in CAT_COLS_RAW: Xr[c] = pd.Categorical(Xr[c], categories=ovocab[c])
    for c in FEATS:
        if c not in CAT_COLS_RAW: Xr[c] = Xr[c].astype("float32")
    p = om.predict_proba(Xr)[:, 1].astype("float32")
    return p, np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).astype("float32")

full = pd.concat([train.drop(columns=["satisfaction"]), test], ignore_index=True)
op, ol = orig_pred(train); opte, olte = orig_pred(test)

# ---- 航线画像（按精确距离均值）+ 值计数（full 上算，fold-invariant）----
coded = full.copy()
for c in CAT_COLS_RAW: coded[c] = coded[c].astype("category").cat.codes
grp = coded.groupby("Flight Distance")
FDAGG = pd.DataFrame({f"fdm_{c}": grp[c].transform("mean").astype(np.float32) for c in FEATS if c != "Flight Distance"})
FDAGG["fd_cnt"] = grp["Age"].transform("size").astype(np.float32)
CNT = pd.DataFrame({f"cnt_{c}": full[c].map(full[c].value_counts()).fillna(0).astype(np.float32) for c in NUM_COLS})

# ---- aux ----
aux_tr = np.load("cache/aux_ev_train.npy"); aux_te = np.load("cache/aux_ev_test.npy")

# ---- 组装 RealMLP 特征帧 ----
def build_frame(df, orig_p, orig_l, fdagg, cnt, aux):
    X = df[FEATS].copy()
    for c in CAT_COLS_RAW: X[c] = X[c].astype(str)          # 类别留字符串给 embedding
    X = pd.concat([X, fdagg.reset_index(drop=True), cnt.reset_index(drop=True)], axis=1)
    X["orig_proba"] = orig_p; X["orig_logit"] = orig_l
    for j, c in enumerate(AUX_COLS): X[c] = aux[:, j]
    # RealMLP 不允许连续列有 NaN（Arrival Delay 有 ~0.04% 缺失），填 -1
    for c in X.columns:
        if X[c].dtype != object: X[c] = X[c].fillna(-1)
    return X

Xtr = build_frame(train, op, ol, FDAGG.iloc[:NTR], CNT.iloc[:NTR], aux_tr)
Xte = build_frame(test, opte, olte, FDAGG.iloc[NTR:], CNT.iloc[NTR:], aux_te)
cat_cols = list(CAT_COLS_RAW)
print(f"RealMLP 特征帧: {Xtr.shape}, 类别列={cat_cols}", flush=True)

# ---- goodpjw2008 REALMLP 配方 ----
from pytabkit import RealMLP_TD_Classifier
REALMLP = dict(
    n_ens=8, n_epochs=6, batch_size=256, use_early_stopping=False,
    lr=0.053, wd=0.0236, sq_mom=0.988, lr_sched="flat_anneal", wd_sched="cos_log_15",
    first_layer_lr_factor=0.25, embedding_size=5, max_one_hot_cat_size=18,
    hidden_sizes=[512, 256, 128], act="silu", p_drop=0.05, p_drop_sched="expm4t",
    plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151, plr_sigma=2.33,
    ls_eps=0.01, ls_eps_sched="sqrt_cos", add_front_scale=False, bias_init_mode="neg-uniform-dynamic-2",
    tfms=["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(NTR); pred = np.zeros(len(test))
for fold, (tr, va) in enumerate(skf.split(Xtr, y)):
    t = time.time()
    m = RealMLP_TD_Classifier(**REALMLP, device="cuda", random_state=42, val_metric_name="1-auc_ovr", verbosity=0)
    m.fit(Xtr.iloc[tr], y[tr], X_val=Xtr.iloc[va], y_val=y[va], cat_col_names=cat_cols)
    oof[va] = m.predict_proba(Xtr.iloc[va])[:, 1]
    pred += m.predict_proba(Xte)[:, 1] / 5
    print(f"fold {fold} done: AUC={roc_auc_score(y[va], oof[va]):.5f} ({time.time()-t:.0f}s)", flush=True)

print(f"\nRealMLP 5折 OOF = {roc_auc_score(y, oof):.5f}")
np.save("cache/oof_realmlp.npy", oof); np.save("cache/pred_realmlp.npy", pred)
print("saved cache/oof_realmlp.npy / pred_realmlp.npy", flush=True)
