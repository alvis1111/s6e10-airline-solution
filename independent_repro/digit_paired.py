"""成对单折 LGBM 对照：强基线(route_profile_aux) vs 强基线 + 整数数字位特征。

协议同 lookup_paired.py（与 lookup 消融分开，独立实验）：
- 强基线 = route_profile_aux 特征集；fold 0（10 折 seed 42），同划分同 seed。
- 内层 80/20 ES 定轮数，外层 val 只评估；只 LGBM 单 seed 42。
- digit 特征：Age/Flight Distance/两延误 4 列，整数取位（个/十/百/千），缺失单独标记；
  只在训练折上识别并剔除常量列。不用公开的浮点公式（整数上恒 9）。
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb

DATA = "data/playground-series-s6e10"
REAL = r"C:\Users\GEM07\.cache\kagglehub\datasets\teejmahal20\airline-passenger-satisfaction\versions\1"

RATING_COLS = ["Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
    "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness"]
CAT_COLS_RAW = ["Gender", "Customer Type", "Type of Travel", "Class"]
ORIGINAL_FEATURES = (["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
                     + RATING_COLS + CAT_COLS_RAW)
AUX_COLS = [f"aux_ev_{c}" for c in RATING_COLS]
NUM_INT = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train["satisfaction"].to_numpy(dtype=np.int64)

# ---- digit 特征（原始值整数取位，缺失保留 NaN） ----
def digit_feats(raw):
    d = {}
    for c in NUM_INT:
        v = raw[c]
        d[f"dg_{c}_u"] = (v % 10).astype("float32")              # 个位
        d[f"dg_{c}_t"] = ((v // 10) % 10).astype("float32")      # 十位
        d[f"dg_{c}_h"] = ((v // 100) % 10).astype("float32")     # 百位
        d[f"dg_{c}_th"] = ((v // 1000) % 10).astype("float32")   # 千位
    d["dg_arrival_missing"] = raw["Arrival Delay in Minutes"].isna().astype("float32")
    return pd.DataFrame(d)

dg_tr = digit_feats(train)
dg_te = digit_feats(test)

# aux 期望值特征
aux_tr = np.load("cache/aux_ev_train.npy"); aux_te = np.load("cache/aux_ev_test.npy")
for j, c in enumerate(AUX_COLS):
    train[c] = aux_tr[:, j]; test[c] = aux_te[:, j]

# 原始数据 teacher 特征
orig_tr = pd.read_csv(f"{REAL}/train.csv"); orig_te = pd.read_csv(f"{REAL}/test.csv")
original = pd.concat([orig_tr, orig_te], ignore_index=True)
original["satisfaction"] = (original["satisfaction"] == "satisfied").astype(int)
original = original.drop(columns=["Unnamed: 0", "id", "Inflight service"], errors="ignore")
def row_hashes(df): return pd.util.hash_pandas_object(df[ORIGINAL_FEATURES].astype(str), index=False)
comp_hashes = pd.concat([row_hashes(train), row_hashes(test)], ignore_index=True)
overlap = row_hashes(original).isin(comp_hashes)
orig_clean = original[~overlap].drop_duplicates().reset_index(drop=True)
orig_y = orig_clean["satisfaction"].astype(int)

_cat_vocab = {}
for col in CAT_COLS_RAW:
    cats = sorted(orig_clean[col].dropna().unique().tolist()); _cat_vocab[col] = cats
    orig_clean[col] = pd.Categorical(orig_clean[col], categories=cats)
for col in ORIGINAL_FEATURES:
    if col not in CAT_COLS_RAW: orig_clean[col] = orig_clean[col].astype("float32")
orig_model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=600, learning_rate=0.05,
    max_depth=8, subsample=0.8, colsample_bytree=0.5, tree_method="hist",
    enable_categorical=True, n_jobs=-1, random_state=0, verbosity=0)
orig_model.fit(orig_clean[ORIGINAL_FEATURES], orig_y)
def apply_orig_model(df):
    Xr = df[ORIGINAL_FEATURES].copy()
    for col in CAT_COLS_RAW: Xr[col] = pd.Categorical(Xr[col], categories=_cat_vocab[col])
    for col in ORIGINAL_FEATURES:
        if col not in CAT_COLS_RAW: Xr[col] = Xr[col].astype("float32")
    p = orig_model.predict_proba(Xr)[:, 1].astype("float32")
    logit = np.log(np.clip(p, 1e-6, 1-1e-6) / (1 - np.clip(p, 1e-6, 1-1e-6))).astype("float32")
    return p, logit
train["orig_proba"], train["orig_logit"] = apply_orig_model(train)
test["orig_proba"], test["orig_logit"] = apply_orig_model(test)

_cat_encode = {"Gender": {"Male": 1, "Female": 0}, "Customer Type": {"Loyal Customer": 1, "disloyal Customer": 0},
    "Type of Travel": {"Business travel": 1, "Personal Travel": 0}, "Class": {"Business": 2, "Eco Plus": 1, "Eco": 0}}
_enc = orig_clean[ORIGINAL_FEATURES].copy()
for col, m in _cat_encode.items(): _enc[col] = orig_clean[col].map(m)
orig_mean_maps = {}
for cols in [("Type of Travel",), ("Class",), ("Customer Type",), ("Type of Travel", "Class", "Customer Type")]:
    key = _enc[list(cols)].astype(str).apply("|".join, axis=1)
    orig_mean_maps[cols] = pd.DataFrame({"key": key, "label": orig_y}).groupby("key")["label"].mean()
orig_prior = float(orig_y.mean())

def base_preprocess(df):
    df = df.copy()
    df.drop(columns=["id"], inplace=True, errors="ignore")
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])
    cat_map = {"Gender": {"Male": 1, "Female": 0}, "Customer Type": {"Loyal Customer": 1, "disloyal Customer": 0},
        "Type of Travel": {"Business travel": 1, "Personal Travel": 0}, "Class": {"Business": 2, "Eco Plus": 1, "Eco": 0}}
    for col, m in cat_map.items(): df[col] = df[col].map(m)
    df["total_delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    df["is_delayed"] = (df["Departure Delay in Minutes"] > 0).astype(int)
    df["avg_rating"] = df[RATING_COLS].mean(axis=1)
    df["min_rating"] = df[RATING_COLS].min(axis=1)
    df["max_rating"] = df[RATING_COLS].max(axis=1)
    df["rating_std"] = df[RATING_COLS].std(axis=1)
    df["zero_count"] = (df[RATING_COLS] == 0).sum(axis=1)
    df["has_zero"] = (df["zero_count"] > 0).astype(int)
    df["age_group"] = pd.cut(df["Age"], bins=[0, 18, 30, 45, 60, 100], labels=[0, 1, 2, 3, 4]).astype(float)
    df["biz_biz"] = ((df["Type of Travel"] == 1) & (df["Class"] == 2)).astype(int)
    df["distance_bin"] = pd.cut(df["Flight Distance"], bins=[0, 500, 1500, float("inf")], labels=[0, 1, 2]).astype(float)
    df["weighted_rating"] = df["Online boarding"] * 0.25 + df["Inflight entertainment"] * 0.20 + df["Seat comfort"] * 0.15 + df[["Inflight wifi service", "On-board service", "Leg room service"]].mean(axis=1) * 0.40
    return df

train_p = base_preprocess(train); test_p = base_preprocess(test)
FEATURE_COLS = [c for c in train_p.columns if c != "satisfaction"]
X_base = train_p[FEATURE_COLS].reset_index(drop=True)
X_test_base = test_p[FEATURE_COLS].reset_index(drop=True)
y = train_p["satisfaction"].reset_index(drop=True)

KEY_PAIRS = [("Class", "Type of Travel"), ("Customer Type", "Type of Travel"), ("Class", "Customer Type"),
    ("Gender", "Class"), ("Inflight wifi service", "Online boarding"), ("Inflight wifi service", "Type of Travel"),
    ("Online boarding", "Type of Travel"), ("Seat comfort", "Inflight entertainment"),
    ("Class", "Inflight wifi service"), ("Class", "Online boarding")]
BASE_ENC_COLS = ["Gender", "Customer Type", "Type of Travel", "Class", "Inflight wifi service",
    "Ease of Online booking", "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Age", "Flight Distance"]

def cat_key(df, cols):
    vals = df[list(cols)].astype(str); key = vals.iloc[:, 0]
    for c in vals.columns[1:]: key = key.str.cat(vals[c], sep="|")
    return key

def add_distance_profile(X_tr, X_val, X_tst):
    dist_col = "Flight Distance"
    measure = RATING_COLS + ["Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
    catp = ["Gender", "Customer Type", "Type of Travel", "Class"]
    grouped = X_tr.groupby(dist_col)[measure].agg(["mean", "std"])
    counts = X_tr[dist_col].value_counts()
    proportions = {col: pd.crosstab(X_tr[dist_col], X_tr[col], normalize="index") for col in catp}
    defaults = {col: {"mean": X_tr[col].mean(), "std": X_tr[col].std()} for col in measure}
    results = []
    for frame in [X_tr, X_val, X_tst]:
        feats = {"profile_route_count": frame[dist_col].map(counts).fillna(0).astype("float32")}
        for col in measure:
            fn = col.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            mu = frame[dist_col].map(grouped[(col, "mean")]).fillna(defaults[col]["mean"]).astype("float32")
            sig = frame[dist_col].map(grouped[(col, "std")]).fillna(defaults[col]["std"]).astype("float32")
            feats[f"profile_{fn}_mean"] = mu; feats[f"profile_{fn}_std"] = sig
            feats[f"profile_{fn}_residual"] = (frame[col] - mu).astype("float32")
        for col, table in proportions.items():
            prior = X_tr[col].value_counts(normalize=True)
            fn = col.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            for cat in table.columns:
                feats[f"profile_{fn}_{cat}_share"] = frame[dist_col].map(table[cat]).fillna(float(prior.get(cat, 0.0))).astype("float32")
        results.append(pd.concat([frame, pd.DataFrame(feats, index=frame.index)], axis=1))
    return results

def add_frequency(X_tr, X_val, X_tst):
    specs = [(c,) for c in BASE_ENC_COLS] + KEY_PAIRS
    results = [X_tr.copy(), X_val.copy(), X_tst.copy()]
    for cols in specs:
        name = "__".join(c.lower().replace(" ", "_").replace("/", "_") for c in cols)
        fm = cat_key(X_tr, cols).value_counts(normalize=True)
        for i, frame in enumerate([X_tr, X_val, X_tst]):
            freq = cat_key(frame, cols).map(fm).fillna(0).astype("float32")
            results[i][f"freq_{name}"] = freq
            results[i][f"rarity_{name}"] = -np.log(freq.clip(lower=1.0 / len(X_tr))).astype("float32")
    return results

def add_orig_means(X_tr, X_val, X_tst):
    results = [X_tr.copy(), X_val.copy(), X_tst.copy()]
    for cols, mapping in orig_mean_maps.items():
        feat = "orig_mean_" + "__".join(c.lower().replace(" ", "_") for c in cols)
        for i, frame in enumerate([X_tr, X_val, X_tst]):
            key = frame[list(cols)].astype(str).apply("|".join, axis=1)
            results[i][feat] = key.map(mapping).fillna(orig_prior).astype("float32")
    return results

def add_te(X_tr, X_val, X_tst, y_tr, fold):
    def build_keys(df):
        keys = {}
        for col in BASE_ENC_COLS: keys[col] = df[col]
        for cols in KEY_PAIRS:
            name = "pair_" + "__".join(c.lower().replace(" ", "_") for c in cols)
            keys[name] = df[cols[0]] * 10 + df[cols[1]] if all(c in RATING_COLS for c in cols) else cat_key(df, cols).astype("category")
        keys["distance_250_bin"] = (df["Flight Distance"] // 250).astype("float32")
        keys["age_5_bin"] = (df["Age"] // 5).astype("float32")
        return pd.DataFrame(keys, index=df.index)
    enc = TargetEncoder(target_type="binary", smooth=20.0, cv=5, shuffle=True, random_state=1042 + fold)
    tr = enc.fit_transform(build_keys(X_tr), y_tr); va = enc.transform(build_keys(X_val)); te = enc.transform(build_keys(X_tst))
    names = [f"te_{c}" for c in build_keys(X_tr).columns]
    return (pd.concat([X_tr, pd.DataFrame(tr.astype("float32"), columns=names, index=X_tr.index)], axis=1),
            pd.concat([X_val, pd.DataFrame(va.astype("float32"), columns=names, index=X_val.index)], axis=1),
            pd.concat([X_tst, pd.DataFrame(te.astype("float32"), columns=names, index=X_tst.index)], axis=1))

lgb_params = dict(objective="binary", metric="auc", n_estimators=4000, learning_rate=0.04,
    num_leaves=63, max_depth=6, subsample=0.9, subsample_freq=1, colsample_bytree=0.85,
    reg_alpha=0.05, reg_lambda=5.0, min_child_samples=20, n_jobs=-1, verbosity=-1)

import sys
FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
skf = StratifiedKFold(10, shuffle=True, random_state=42)
tr, va = list(skf.split(X_base, y))[FOLD]
print(f"fold {FOLD}: train {len(tr)} / val {len(va)}", flush=True)

X_tr, X_val, X_tst = add_distance_profile(X_base.iloc[tr], X_base.iloc[va], X_test_base)
X_tr, X_val, X_tst = add_frequency(X_tr, X_val, X_tst)
X_tr, X_val, X_tst = add_orig_means(X_tr, X_val, X_tst)
X_tr, X_val, X_tst = add_te(X_tr, X_val, X_tst, y.iloc[tr], 0)
for f in [X_tr, X_val, X_tst]:
    f.replace([np.inf, -np.inf], 0, inplace=True); f.fillna(0, inplace=True)
    f = f.loc[:, ~f.columns.duplicated()].astype("float32")
X_tr = X_tr[[c for c in X_tr.columns if X_tr[c].nunique() > 1]]
X_val = X_val[X_tr.columns]; X_tst = X_tst[X_tr.columns]
print(f"强基线特征数: {X_tr.shape[1]}", flush=True)

# digit 有效列（只在训练折上剔常量）
dg_fold = dg_tr.iloc[tr]
dg_cols = [c for c in dg_fold.columns if dg_fold[c].nunique() > 1]
print(f"digit 有效列: {len(dg_cols)} / {dg_tr.shape[1]}  -> {dg_cols}", flush=True)
X_tr_d = pd.concat([X_tr, dg_fold[dg_cols].astype("float32")], axis=1)
X_val_d = pd.concat([X_val, dg_tr.iloc[va][dg_cols].astype("float32")], axis=1)


def paired_auc(Xtr, Xval, tag):
    inner = StratifiedKFold(5, shuffle=True, random_state=0)
    i_tr, i_va = next(iter(inner.split(Xtr, y.iloc[tr])))
    m = lgb.LGBMClassifier(**{**lgb_params, "random_state": 42})
    m.fit(Xtr.iloc[i_tr], y.iloc[tr].iloc[i_tr], eval_set=[(Xtr.iloc[i_va], y.iloc[tr].iloc[i_va])],
          callbacks=[lgb.early_stopping(150, verbose=False)])
    best_iter = m.best_iteration_ or lgb_params["n_estimators"]
    m2 = lgb.LGBMClassifier(**{**lgb_params, "n_estimators": best_iter, "random_state": 42})
    m2.fit(Xtr, y.iloc[tr])
    auc = roc_auc_score(y.iloc[va], m2.predict_proba(Xval)[:, 1])
    print(f"{tag}: 内层 best_iter={best_iter}  外层 val AUC={auc:.6f}", flush=True)
    return auc, best_iter


auc_base, it_base = paired_auc(X_tr, X_val, "强基线     ")
auc_cand, it_cand = paired_auc(X_tr_d, X_val_d, "强基线+digit")
print(f"\nΔAUC = {auc_cand - auc_base:+.6f}  (候选 {auc_cand:.6f} vs 基线 {auc_base:.6f})", flush=True)
