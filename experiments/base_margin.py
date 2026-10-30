"""S6E10 — 纠偏实验：把原始模型预测从「特征」改成「base_margin / init_score」。

现状：orig_logit/orig_proba 当普通特征喂给模型（+0.00056，是王炸特征）。
新做法：用 orig_logit 当 base_margin（XGBoost）/ init_score（LightGBM），强制模型从
sigmoid(orig_logit) 出发，树只学「哪些人群被高估/低估」的残差纠偏，线性部分免费送。

对比（5 折单种子，route_profile 全套特征）：
  A. XGBoost + orig 当特征（现状基线）
  B. XGBoost + base_margin=orig_logit（删掉 orig 特征列）
  C. LightGBM + init_score=orig_logit（删掉 orig 特征列）
参考：sigmoid(orig_logit) 本身的 AUC（base 的起点）。
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
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
ORIGINAL_FEATURES = NUM_COLS + RATING_COLS + CAT_COLS_RAW
ORIG_PRED_COLS = ["orig_proba", "orig_logit"]

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train["satisfaction"].to_numpy(dtype=np.int64)

# ---- 原始数据 + orig_model + orig_means（同 route_profile.py）----
orig_tr = pd.read_csv(f"{REAL}/train.csv"); orig_te = pd.read_csv(f"{REAL}/test.csv")
original = pd.concat([orig_tr, orig_te], ignore_index=True)
original["satisfaction"] = (original["satisfaction"] == "satisfied").astype(int)
original = original.drop(columns=["Unnamed: 0", "id", "Inflight service"], errors="ignore")
def row_hashes(df):
    return pd.util.hash_pandas_object(df[ORIGINAL_FEATURES].astype(str), index=False)
comp_hashes = pd.concat([row_hashes(train), row_hashes(test)], ignore_index=True)
overlap = row_hashes(original).isin(comp_hashes)
orig_clean = original[~overlap].drop_duplicates().reset_index(drop=True)
orig_y = orig_clean["satisfaction"].astype(int)

_cat_vocab = {}
for col in CAT_COLS_RAW:
    cats = sorted(orig_clean[col].dropna().unique().tolist())
    _cat_vocab[col] = cats
    orig_clean[col] = pd.Categorical(orig_clean[col], categories=cats)
_orig_X = orig_clean[ORIGINAL_FEATURES].copy()
for col in ORIGINAL_FEATURES:
    if col not in CAT_COLS_RAW:
        _orig_X[col] = _orig_X[col].astype("float32")
orig_model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=600, learning_rate=0.05,
    max_depth=8, subsample=0.8, colsample_bytree=0.5, tree_method="hist",
    enable_categorical=True, n_jobs=-1, random_state=0, verbosity=0).fit(_orig_X, orig_y)

def apply_orig_model(df):
    Xr = df[ORIGINAL_FEATURES].copy()
    for col in CAT_COLS_RAW:
        Xr[col] = pd.Categorical(Xr[col], categories=_cat_vocab[col])
    for col in ORIGINAL_FEATURES:
        if col not in CAT_COLS_RAW:
            Xr[col] = Xr[col].astype("float32")
    p = orig_model.predict_proba(Xr)[:, 1].astype("float32")
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).astype("float32")
    return p, logit

train["orig_proba"], train["orig_logit"] = apply_orig_model(train)
test["orig_proba"], test["orig_logit"] = apply_orig_model(test)
# 记住 raw orig_logit（base_margin 用，行序与 base_preprocess 后一致）
base_logit_tr = train["orig_logit"].to_numpy().copy()
base_logit_te = test["orig_logit"].to_numpy().copy()

_cat_encode = {"Gender": {"Male": 1, "Female": 0}, "Customer Type": {"Loyal Customer": 1, "disloyal Customer": 0},
    "Type of Travel": {"Business travel": 1, "Personal Travel": 0}, "Class": {"Business": 2, "Eco Plus": 1, "Eco": 0}}
_enc = orig_clean[ORIGINAL_FEATURES].copy()
for col, m in _cat_encode.items():
    _enc[col] = orig_clean[col].map(m)
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
    for col, m in cat_map.items():
        df[col] = df[col].map(m)
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
    for c in vals.columns[1:]:
        key = key.str.cat(vals[c], sep="|")
    return key

def add_distance_profile(X_tr, X_val, X_tst):
    dist_col = "Flight Distance"; measure = RATING_COLS + ["Age", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
    catp = ["Gender", "Customer Type", "Type of Travel", "Class"]
    grouped = X_tr.groupby(dist_col)[measure].agg(["mean", "std"]); counts = X_tr[dist_col].value_counts()
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
            prior = X_tr[col].value_counts(normalize=True); fn = col.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
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
xgb_params = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist",
    n_estimators=4000, learning_rate=0.04, max_depth=6, min_child_weight=5,
    subsample=0.9, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=5.0,
    max_bin=256, early_stopping_rounds=150, n_jobs=-1, verbosity=0)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = {"A_xgb_feat": np.zeros(len(X_base)), "B_xgb_margin": np.zeros(len(X_base)), "C_lgb_init": np.zeros(len(X_base))}

for fold, (tr, va) in enumerate(skf.split(X_base, y)):
    t = time.time()
    X_tr, X_val, X_tst = add_distance_profile(X_base.iloc[tr], X_base.iloc[va], X_test_base)
    X_tr, X_val, X_tst = add_frequency(X_tr, X_val, X_tst)
    X_tr, X_val, X_tst = add_orig_means(X_tr, X_val, X_tst)
    X_tr, X_val, X_tst = add_te(X_tr, X_val, X_tst, y.iloc[tr], fold)
    for f in [X_tr, X_val, X_tst]:
        f.replace([np.inf, -np.inf], 0, inplace=True); f.fillna(0, inplace=True)
        f = f.loc[:, ~f.columns.duplicated()].astype("float32")
    X_tr = X_tr[[c for c in X_tr.columns if X_tr[c].nunique() > 1]]
    X_val = X_val[X_tr.columns]
    feats_all = X_tr.columns.tolist()
    feats_noorig = [c for c in feats_all if c not in ORIG_PRED_COLS]
    mg_tr = base_logit_tr[tr]; mg_va = base_logit_tr[va]

    # A. XGBoost, orig 当特征（现状）
    ma = xgb.XGBClassifier(**{**xgb_params, "random_state": 42})
    ma.fit(X_tr[feats_all], y.iloc[tr], eval_set=[(X_val[feats_all], y.iloc[va])], verbose=False)
    oof["A_xgb_feat"][va] = ma.predict_proba(X_val[feats_all])[:, 1]

    # B. XGBoost, base_margin=orig_logit（删 orig 特征）
    mb = xgb.XGBClassifier(**{**xgb_params, "random_state": 42})
    mb.fit(X_tr[feats_noorig], y.iloc[tr], base_margin=mg_tr,
           eval_set=[(X_val[feats_noorig], y.iloc[va])], base_margin_eval_set=[mg_va], verbose=False)
    oof["B_xgb_margin"][va] = mb.predict_proba(X_val[feats_noorig], base_margin=mg_va)[:, 1]

    # C. LightGBM, init_score=orig_logit（删 orig 特征）
    mc = lgb.LGBMClassifier(**{**lgb_params, "random_state": 42})
    mc.fit(X_tr[feats_noorig], y.iloc[tr], init_score=mg_tr,
           eval_set=[(X_val[feats_noorig], y.iloc[va])], eval_init_score=[mg_va],
           callbacks=[lgb.early_stopping(150, verbose=False)])
    oof["C_lgb_init"][va] = mc.predict_proba(X_val[feats_noorig])[:, 1]

    print(f"fold {fold}: A_xgb_feat={roc_auc_score(y.iloc[va], oof['A_xgb_feat'][va]):.5f} "
          f"B_xgb_margin={roc_auc_score(y.iloc[va], oof['B_xgb_margin'][va]):.5f} "
          f"C_lgb_init={roc_auc_score(y.iloc[va], oof['C_lgb_init'][va]):.5f}  ({time.time()-t:.0f}s)", flush=True)

print("\n" + "=" * 62)
print(f"  sigmoid(orig_logit) 自身 AUC = {roc_auc_score(y, 1/(1+np.exp(-base_logit_tr))):.5f}")
for k in ["A_xgb_feat", "B_xgb_margin", "C_lgb_init"]:
    print(f"  {k:14s} OOF = {roc_auc_score(y, oof[k]):.5f}")
print("=" * 62)
np.save("cache/oof_base_margin.npz", **oof)
print("saved cache/oof_base_margin.npz", flush=True)
