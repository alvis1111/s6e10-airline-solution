"""S6E10 — 航线画像(精确距离) + 原始数据特征 + 精确距离TE. 单 LGBM 验证."""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

DATA = "data/playground-series-s6e10"
REAL = r"C:\Users\GEM07\.cache\kagglehub\datasets\teejmahal20\airline-passenger-satisfaction\versions\1"

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train['satisfaction'].to_numpy(dtype=np.int64)

RATING_COLS = ["Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
    "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness"]
CAT_COLS_RAW = ["Gender", "Customer Type", "Type of Travel", "Class"]
ORIGINAL_FEATURES = [c for c in train.columns if c not in ["id", "satisfaction"]]

# ---- 原始数据 (teejmahal20 = 129k, 与 notebook 的 129k 同源) ----
orig_tr = pd.read_csv(f"{REAL}/train.csv")
orig_te = pd.read_csv(f"{REAL}/test.csv")
original = pd.concat([orig_tr, orig_te], ignore_index=True)
original["satisfaction"] = (original["satisfaction"] == "satisfied").astype(int)
original = original.drop(columns=["Unnamed: 0", "id", "Inflight service"], errors="ignore")
# 去重 + 去重叠
def row_hashes(df):
    return pd.util.hash_pandas_object(df[ORIGINAL_FEATURES].astype(str), index=False)
comp_hashes = pd.concat([row_hashes(train), row_hashes(test)], ignore_index=True)
orig_hashes = row_hashes(original)
overlap = orig_hashes.isin(comp_hashes)
orig_clean = original[~overlap].drop_duplicates().reset_index(drop=True)
orig_y = orig_clean["satisfaction"].astype(int)
orig_X = orig_clean[ORIGINAL_FEATURES].copy()
print(f"原始数据: {len(original)} -> 去重叠后 {len(orig_clean)}", flush=True)

# 原始数据上训 XGBoost 当特征生成器
_cat_vocab = {}
for col in CAT_COLS_RAW:
    cats = sorted(orig_X[col].dropna().unique().tolist())
    _cat_vocab[col] = cats
    orig_X[col] = pd.Categorical(orig_X[col], categories=cats)
for col in ORIGINAL_FEATURES:
    if col not in CAT_COLS_RAW:
        orig_X[col] = orig_X[col].astype("float32")
orig_model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=600, learning_rate=0.05,
    max_depth=8, subsample=0.8, colsample_bytree=0.5, tree_method="hist",
    enable_categorical=True, n_jobs=-1, random_state=0, verbosity=0)
orig_model.fit(orig_X, orig_y)

def apply_orig_model(df):
    Xr = df[ORIGINAL_FEATURES].copy()
    for col in CAT_COLS_RAW:
        Xr[col] = pd.Categorical(Xr[col], categories=_cat_vocab[col])
    for col in ORIGINAL_FEATURES:
        if col not in CAT_COLS_RAW:
            Xr[col] = Xr[col].astype("float32")
    p = orig_model.predict_proba(Xr)[:, 1].astype("float32")
    logit = np.log(np.clip(p, 1e-6, 1-1e-6)/(1-np.clip(p, 1e-6, 1-1e-6))).astype("float32")
    return p, logit

train["orig_proba"], train["orig_logit"] = apply_orig_model(train)
test["orig_proba"], test["orig_logit"] = apply_orig_model(test)

# 原始目标均值
_cat_encode = {"Gender": {"Male":1,"Female":0}, "Customer Type": {"Loyal Customer":1,"disloyal Customer":0},
    "Type of Travel": {"Business travel":1,"Personal Travel":0}, "Class": {"Business":2,"Eco Plus":1,"Eco":0}}
_enc = orig_clean[ORIGINAL_FEATURES].copy()
for col, m in _cat_encode.items(): _enc[col] = orig_clean[col].map(m)
orig_mean_maps = {}
for cols in [("Type of Travel",),("Class",),("Customer Type",),("Type of Travel","Class","Customer Type")]:
    key = _enc[list(cols)].astype(str).apply("|".join, axis=1)
    orig_mean_maps[cols] = pd.DataFrame({"key":key,"label":orig_y}).groupby("key")["label"].mean()
orig_prior = float(orig_y.mean())

# ---- base preprocess ----
def base_preprocess(df):
    df = df.copy()
    df.drop(columns=["id"], inplace=True, errors="ignore")
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])
    cat_map = {"Gender": {"Male":1,"Female":0}, "Customer Type": {"Loyal Customer":1,"disloyal Customer":0},
        "Type of Travel": {"Business travel":1,"Personal Travel":0}, "Class": {"Business":2,"Eco Plus":1,"Eco":0}}
    for col, m in cat_map.items(): df[col] = df[col].map(m)
    df["total_delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    df["is_delayed"] = (df["Departure Delay in Minutes"] > 0).astype(int)
    df["avg_rating"] = df[RATING_COLS].mean(axis=1)
    df["min_rating"] = df[RATING_COLS].min(axis=1)
    df["max_rating"] = df[RATING_COLS].max(axis=1)
    df["rating_std"] = df[RATING_COLS].std(axis=1)
    df["zero_count"] = (df[RATING_COLS] == 0).sum(axis=1)
    df["has_zero"] = (df["zero_count"] > 0).astype(int)
    df["age_group"] = pd.cut(df["Age"], bins=[0,18,30,45,60,100], labels=[0,1,2,3,4]).astype(float)
    df["biz_biz"] = ((df["Type of Travel"]==1)&(df["Class"]==2)).astype(int)
    df["distance_bin"] = pd.cut(df["Flight Distance"], bins=[0,500,1500,float("inf")], labels=[0,1,2]).astype(float)
    df["weighted_rating"] = df["Online boarding"]*0.25 + df["Inflight entertainment"]*0.20 + df["Seat comfort"]*0.15 + df[["Inflight wifi service","On-board service","Leg room service"]].mean(axis=1)*0.40
    return df

train_p = base_preprocess(train); test_p = base_preprocess(test)
FEATURE_COLS = [c for c in train_p.columns if c != "satisfaction"]
X_base = train_p[FEATURE_COLS].reset_index(drop=True)
X_test_base = test_p[FEATURE_COLS].reset_index(drop=True)
y = train_p["satisfaction"].reset_index(drop=True)

# ---- fold-level features ----
KEY_PAIRS = [("Class","Type of Travel"),("Customer Type","Type of Travel"),("Class","Customer Type"),
    ("Gender","Class"),("Inflight wifi service","Online boarding"),("Inflight wifi service","Type of Travel"),
    ("Online boarding","Type of Travel"),("Seat comfort","Inflight entertainment"),
    ("Class","Inflight wifi service"),("Class","Online boarding")]
BASE_ENC_COLS = ["Gender","Customer Type","Type of Travel","Class","Inflight wifi service",
    "Ease of Online booking","Online boarding","Seat comfort","Inflight entertainment",
    "On-board service","Age","Flight Distance"]

def cat_key(df, cols):
    vals = df[list(cols)].astype(str)
    key = vals.iloc[:,0]
    for c in vals.columns[1:]: key = key.str.cat(vals[c], sep="|")
    return key

def add_distance_profile(X_tr, X_val, X_tst):
    dist_col = "Flight Distance"
    measure = RATING_COLS + ["Age","Departure Delay in Minutes","Arrival Delay in Minutes"]
    catp = ["Gender","Customer Type","Type of Travel","Class"]
    grouped = X_tr.groupby(dist_col)[measure].agg(["mean","std"])
    counts = X_tr[dist_col].value_counts()
    proportions = {col: pd.crosstab(X_tr[dist_col], X_tr[col], normalize="index") for col in catp}
    defaults = {col: {"mean": X_tr[col].mean(), "std": X_tr[col].std()} for col in measure}
    results = []
    for frame in [X_tr, X_val, X_tst]:
        feats = {"profile_route_count": frame[dist_col].map(counts).fillna(0).astype("float32")}
        for col in measure:
            fn = col.lower().replace(" ","_").replace("/","_").replace("-","_")
            mu = frame[dist_col].map(grouped[(col,"mean")]).fillna(defaults[col]["mean"]).astype("float32")
            sig = frame[dist_col].map(grouped[(col,"std")]).fillna(defaults[col]["std"]).astype("float32")
            feats[f"profile_{fn}_mean"] = mu
            feats[f"profile_{fn}_std"] = sig
            feats[f"profile_{fn}_residual"] = (frame[col] - mu).astype("float32")
        for col, table in proportions.items():
            prior = X_tr[col].value_counts(normalize=True)
            fn = col.lower().replace(" ","_").replace("/","_").replace("-","_")
            for cat in table.columns:
                feats[f"profile_{fn}_{cat}_share"] = frame[dist_col].map(table[cat]).fillna(float(prior.get(cat,0.0))).astype("float32")
        results.append(pd.concat([frame, pd.DataFrame(feats, index=frame.index)], axis=1))
    return results

def add_frequency(X_tr, X_val, X_tst):
    specs = [(c,) for c in BASE_ENC_COLS] + KEY_PAIRS
    results = [X_tr.copy(), X_val.copy(), X_tst.copy()]
    for cols in specs:
        name = "__".join(c.lower().replace(" ","_").replace("/","_") for c in cols)
        fm = cat_key(X_tr, cols).value_counts(normalize=True)
        for i, frame in enumerate([X_tr, X_val, X_tst]):
            freq = cat_key(frame, cols).map(fm).fillna(0).astype("float32")
            results[i][f"freq_{name}"] = freq
            results[i][f"rarity_{name}"] = -np.log(freq.clip(lower=1.0/len(X_tr))).astype("float32")
    return results

def add_orig_means(X_tr, X_val, X_tst):
    results = [X_tr.copy(), X_val.copy(), X_tst.copy()]
    for cols, mapping in orig_mean_maps.items():
        feat = "orig_mean_" + "__".join(c.lower().replace(" ","_") for c in cols)
        for i, frame in enumerate([X_tr, X_val, X_tst]):
            key = frame[list(cols)].astype(str).apply("|".join, axis=1)
            results[i][feat] = key.map(mapping).fillna(orig_prior).astype("float32")
    return results

def add_te(X_tr, X_val, X_tst, y_tr, fold):
    def build_keys(df):
        keys = {}
        for col in BASE_ENC_COLS: keys[col] = df[col]
        for cols in KEY_PAIRS:
            name = "pair_" + "__".join(c.lower().replace(" ","_") for c in cols)
            keys[name] = df[cols[0]]*10 + df[cols[1]] if all(c in RATING_COLS for c in cols) else cat_key(df, cols).astype("category")
        keys["distance_250_bin"] = (df["Flight Distance"]//250).astype("float32")
        keys["age_5_bin"] = (df["Age"]//5).astype("float32")
        return pd.DataFrame(keys, index=df.index)
    enc = TargetEncoder(target_type="binary", smooth=20.0, cv=5, shuffle=True, random_state=1042+fold)
    tr = enc.fit_transform(build_keys(X_tr), y_tr)
    va = enc.transform(build_keys(X_val))
    te = enc.transform(build_keys(X_tst))
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
SEEDS = [42, 7, 2024]

skf = StratifiedKFold(10, shuffle=True, random_state=42)
oof_l = np.zeros(len(X_base)); oof_x = np.zeros(len(X_base))
pred_l = np.zeros(len(X_test_base)); pred_x = np.zeros(len(X_test_base))

for fold, (tr, va) in enumerate(skf.split(X_base, y)):
    X_tr, X_val, X_tst = add_distance_profile(X_base.iloc[tr], X_base.iloc[va], X_test_base)
    X_tr, X_val, X_tst = add_frequency(X_tr, X_val, X_tst)
    X_tr, X_val, X_tst = add_orig_means(X_tr, X_val, X_tst)
    X_tr, X_val, X_tst = add_te(X_tr, X_val, X_tst, y.iloc[tr], fold)
    for f in [X_tr, X_val, X_tst]:
        f.replace([np.inf,-np.inf], 0, inplace=True); f.fillna(0, inplace=True)
        f = f.loc[:, ~f.columns.duplicated()].astype("float32")
    X_tr = X_tr[[c for c in X_tr.columns if X_tr[c].nunique()>1]]
    X_val = X_val[X_tr.columns]; X_tst = X_tst[X_tr.columns]
    if fold == 0: print(f"特征数: {X_tr.shape[1]}", flush=True)
    for s in SEEDS:
        ml = lgb.LGBMClassifier(**{**lgb_params, "random_state": s})
        ml.fit(X_tr, y.iloc[tr], eval_set=[(X_val, y.iloc[va])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof_l[va] += ml.predict_proba(X_val)[:,1]/len(SEEDS)
        pred_l += ml.predict_proba(X_tst)[:,1]/(10*len(SEEDS))
        mx = xgb.XGBClassifier(**{**xgb_params, "random_state": s})
        mx.fit(X_tr, y.iloc[tr], eval_set=[(X_val, y.iloc[va])], verbose=False)
        oof_x[va] += mx.predict_proba(X_val)[:,1]/len(SEEDS)
        pred_x += mx.predict_proba(X_tst)[:,1]/(10*len(SEEDS))
    print(f"fold {fold} done: LGB={roc_auc_score(y.iloc[va], oof_l[va]):.5f} XGB={roc_auc_score(y.iloc[va], oof_x[va]):.5f}", flush=True)

blend_oof = 0.5*oof_l + 0.5*oof_x
blend_pred = 0.5*pred_l + 0.5*pred_x
print(f"\nLGBM(3seed) OOF = {roc_auc_score(y, oof_l):.5f}", flush=True)
print(f"XGB(3seed)  OOF = {roc_auc_score(y, oof_x):.5f}", flush=True)
print(f"Blend        OOF = {roc_auc_score(y, blend_oof):.5f}", flush=True)
np.save("cache/oof_route_l.npy", oof_l); np.save("cache/pred_route_l.npy", pred_l)
np.save("cache/oof_route_x.npy", oof_x); np.save("cache/pred_route_x.npy", pred_x)
np.save("cache/pred_route_blend.npy", blend_pred)
print("saved", flush=True)
