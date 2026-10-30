"""S6E10 — RealMLP 完整复现：加 fold 内目标编码(TE)，对齐 goodpjw2008 的 view-A 输入。

= realmlp_repro.py 的 base 特征(raw + 航线画像 + 计数 + orig + aux) + 每折 fold-safe TE(24列)。
TE 用内层 cv=5，训练行不泄漏。5 折 seed 42。
"""
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
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

BASE_ENC_COLS = ["Gender", "Customer Type", "Type of Travel", "Class", "Inflight wifi service",
    "Ease of Online booking", "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Age", "Flight Distance"]
KEY_PAIRS = [("Class", "Type of Travel"), ("Customer Type", "Type of Travel"), ("Class", "Customer Type"),
    ("Gender", "Class"), ("Inflight wifi service", "Online boarding"), ("Inflight wifi service", "Type of Travel"),
    ("Online boarding", "Type of Travel"), ("Seat comfort", "Inflight entertainment"),
    ("Class", "Inflight wifi service"), ("Class", "Online boarding")]

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train["satisfaction"].to_numpy(dtype=np.int64)
NTR = len(train)

# ---- orig 模型 ----
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

# ---- 航线画像 + 值计数 ----
coded = full.copy()
for c in CAT_COLS_RAW: coded[c] = coded[c].astype("category").cat.codes
grp = coded.groupby("Flight Distance")
FDAGG = pd.DataFrame({f"fdm_{c}": grp[c].transform("mean").astype(np.float32) for c in FEATS if c != "Flight Distance"})
FDAGG["fd_cnt"] = grp["Age"].transform("size").astype(np.float32)
CNT = pd.DataFrame({f"cnt_{c}": full[c].map(full[c].value_counts()).fillna(0).astype(np.float32) for c in NUM_COLS})

aux_tr = np.load("cache/aux_ev_train.npy"); aux_te = np.load("cache/aux_ev_test.npy")

def build_base(df, orig_p, orig_l, fdagg, cnt, aux):
    X = df[FEATS].copy()
    for c in CAT_COLS_RAW: X[c] = X[c].astype(str)
    # 数值列类别副本（公开配方：保留数值列同时增加类别副本，交给 embedding/one-hot）
    for c in NUM_COLS + RATING_COLS:
        X[c + "_cat"] = df[c].astype(str)
    X = pd.concat([X, fdagg.reset_index(drop=True), cnt.reset_index(drop=True)], axis=1)
    X["orig_proba"] = orig_p; X["orig_logit"] = orig_l
    for j, c in enumerate(AUX_COLS): X[c] = aux[:, j]
    for c in X.columns:
        if X[c].dtype != object: X[c] = X[c].fillna(-1)
    return X

Xtr_base = build_base(train, op, ol, FDAGG.iloc[:NTR], CNT.iloc[:NTR], aux_tr)
Xte_base = build_base(test, opte, olte, FDAGG.iloc[NTR:], CNT.iloc[NTR:], aux_te)
cat_cols = list(CAT_COLS_RAW) + [c + "_cat" for c in NUM_COLS + RATING_COLS]

# ---- TE keys（在 raw train/test 上算，fold-safe）----
raw_tr = train.drop(columns=["satisfaction"]).reset_index(drop=True)
raw_te = test.reset_index(drop=True)

def cat_key(df, cols):
    vals = df[list(cols)].astype(str); key = vals.iloc[:, 0]
    for c in vals.columns[1:]: key = key.str.cat(vals[c], sep="|")
    return key

def build_te_keys(df):
    keys = {}
    for col in BASE_ENC_COLS: keys[col] = df[col]
    for cols in KEY_PAIRS:
        name = "pair_" + "__".join(c.lower().replace(" ", "_") for c in cols)
        keys[name] = df[cols[0]] * 10 + df[cols[1]] if all(c in RATING_COLS for c in cols) else cat_key(df, cols).astype("category")
    keys["distance_250_bin"] = (df["Flight Distance"] // 250).astype("float32")
    keys["age_5_bin"] = (df["Age"] // 5).astype("float32")
    return pd.DataFrame(keys, index=df.index)

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
for fold, (tr, va) in enumerate(skf.split(Xtr_base, y)):
    t = time.time()
    # fold-safe TE
    enc = TargetEncoder(target_type="binary", smooth=20.0, cv=5, shuffle=True, random_state=1042 + fold)
    te_tr = enc.fit_transform(build_te_keys(raw_tr.iloc[tr]), y[tr])
    te_va = enc.transform(build_te_keys(raw_tr.iloc[va]))
    te_te = enc.transform(build_te_keys(raw_te))
    te_names = [f"te_{i}" for i in range(te_tr.shape[1])]
    Xtr = pd.concat([Xtr_base.iloc[tr].reset_index(drop=True),
                     pd.DataFrame(te_tr.astype("float32"), columns=te_names)], axis=1)
    Xva = pd.concat([Xtr_base.iloc[va].reset_index(drop=True),
                     pd.DataFrame(te_va.astype("float32"), columns=te_names)], axis=1)
    Xte = pd.concat([Xte_base.reset_index(drop=True),
                     pd.DataFrame(te_te.astype("float32"), columns=te_names)], axis=1)
    m = RealMLP_TD_Classifier(**REALMLP, device="cuda", random_state=42, val_metric_name="1-auc_ovr", verbosity=0)
    m.fit(Xtr, y[tr], X_val=Xva, y_val=y[va], cat_col_names=cat_cols)
    oof[va] = m.predict_proba(Xva)[:, 1]
    pred += m.predict_proba(Xte)[:, 1] / 5
    print(f"fold {fold} done: AUC={roc_auc_score(y[va], oof[va]):.5f} ({time.time()-t:.0f}s)", flush=True)

print(f"\nRealMLP(含TE) 5折 OOF = {roc_auc_score(y, oof):.5f}")
np.save("cache/oof_realmlp_te.npy", oof); np.save("cache/pred_realmlp_te.npy", pred)
print("saved cache/oof_realmlp_te.npy / pred_realmlp_te.npy", flush=True)
