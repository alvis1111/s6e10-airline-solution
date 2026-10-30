"""S6E10 — 特征家族消融 + 训练配置快查。

背景：notebook 作者自己都说「从没单独删掉某一族特征做过消融」，而我们的 memory 里
「特征穷举无效」只覆盖了聚合/交互类，没覆盖「现有这 5 族特征到底各自值多少」。
本脚本用 5 折单种子 LGBM（快速配置），在同一个 fold 循环里训 6 个模型：
  全特征 / 删航线画像 / 删TE / 删频率 / 删原始均值 / 删原始数据模型概率，
看每族特征的边际贡献，顺便验证训练配置里几个关键超参有没有用。

注意：这是快速配置（5折单种子LGBM），绝对 OOF 会略低于完整版（10折3种子LGBM+XGB 0.96123），
我们看的是**相对 delta**，方向/量级对决策足够。
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

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train["satisfaction"].to_numpy(dtype=np.int64)

# ---- 原始数据（去重叠 + orig_model + orig_means，同 route_profile.py）----
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

# ---- fold-level 特征（同 route_profile.py）----
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

# 消融配置：每族特征按列名前缀选出来，drop 即去掉
FAMILIES = {
    "profile": lambda c: c.startswith("profile_"),
    "te":       lambda c: c.startswith("te_"),
    "freq":     lambda c: c.startswith("freq_") or c.startswith("rarity_"),
    "orig_mean":lambda c: c.startswith("orig_mean_"),
    "orig_prob":lambda c: c in ("orig_proba", "orig_logit"),
}
CONFIGS = {"all": []}
for fam in FAMILIES:
    CONFIGS[f"no_{fam}"] = [fam]

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oofs = {k: np.zeros(len(X_base)) for k in CONFIGS}

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
    all_cols = X_tr.columns.tolist()
    if fold == 0:
        for fam, pred in FAMILIES.items():
            n = sum(1 for c in all_cols if pred(c))
            print(f"  {fam:12s} 特征数 = {n}", flush=True)
    for cfg, drop_fams in CONFIGS.items():
        drop = [fam for fam in drop_fams]
        keep = [c for c in all_cols if not any(FAMILIES[f](c) for f in drop)]
        m = lgb.LGBMClassifier(**{**lgb_params, "random_state": 42})
        m.fit(X_tr[keep], y.iloc[tr], eval_set=[(X_val[keep], y.iloc[va])],
              callbacks=[lgb.early_stopping(150, verbose=False)])
        oofs[cfg][va] = m.predict_proba(X_val[keep])[:, 1]
    print(f"fold {fold} done ({time.time()-t:.0f}s)", flush=True)

print("\n" + "=" * 62)
print("  特征家族消融 (5折单种子 LGBM) OOF")
auc_all = roc_auc_score(y, oofs["all"])
print(f"  {'all':12s} {auc_all:.5f}  (基准)")
for cfg in CONFIGS:
    if cfg == "all": continue
    a = roc_auc_score(y, oofs[cfg])
    print(f"  {cfg:12s} {a:.5f}  ({a-auc_all:+.5f})")
print("=" * 62)
