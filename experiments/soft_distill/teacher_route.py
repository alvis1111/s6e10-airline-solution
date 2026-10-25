"""teacher_route — route 腿的教师 OOF(fold 1-9 上嵌套 5 折 CV, 排除 fold 0)。

输出 cache/teacher_route_oof.npy (只对 ia 行有值, 其余 NaN)。
"""
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

RATING_COLS = ["Inflight wifi service","Departure/Arrival time convenient","Ease of Online booking",
    "Gate location","Food and drink","Online boarding","Seat comfort","Inflight entertainment",
    "On-board service","Leg room service","Baggage handling","Checkin service","Cleanliness"]
CAT_COLS_RAW = ["Gender","Customer Type","Type of Travel","Class"]
ORIGINAL_FEATURES = [c for c in train.columns if c not in ["id","satisfaction"]]

orig_tr = pd.read_csv(f"{REAL}/train.csv"); orig_te = pd.read_csv(f"{REAL}/test.csv")
original = pd.concat([orig_tr, orig_te], ignore_index=True)
original["satisfaction"] = (original["satisfaction"] == "satisfied").astype(int)
original = original.drop(columns=["Unnamed: 0","id","Inflight service"], errors="ignore")
def row_hashes(df): return pd.util.hash_pandas_object(df[ORIGINAL_FEATURES].astype(str), index=False)
comp_hashes = pd.concat([row_hashes(train), row_hashes(test)], ignore_index=True)
overlap = row_hashes(original).isin(comp_hashes)
orig_clean = original[~overlap].drop_duplicates().reset_index(drop=True)
orig_y = orig_clean["satisfaction"].astype(int)
orig_X = orig_clean[ORIGINAL_FEATURES].copy()
_cat_vocab = {}
for col in CAT_COLS_RAW:
    cats = sorted(orig_X[col].dropna().unique().tolist()); _cat_vocab[col] = cats
    orig_X[col] = pd.Categorical(orig_X[col], categories=cats)
for col in ORIGINAL_FEATURES:
    if col not in CAT_COLS_RAW: orig_X[col] = orig_X[col].astype("float32")
orig_model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=600, learning_rate=0.05,
    max_depth=8, subsample=0.8, colsample_bytree=0.5, tree_method="hist", enable_categorical=True,
    n_jobs=-1, random_state=0, verbosity=0)
orig_model.fit(orig_X, orig_y)
def apply_orig_model(df):
    Xr = df[ORIGINAL_FEATURES].copy()
    for col in CAT_COLS_RAW: Xr[col] = pd.Categorical(Xr[col], categories=_cat_vocab[col])
    for col in ORIGINAL_FEATURES:
        if col not in CAT_COLS_RAW: Xr[col] = Xr[col].astype("float32")
    p = orig_model.predict_proba(Xr)[:,1].astype("float32")
    return p, np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).astype("float32")
train["orig_proba"], train["orig_logit"] = apply_orig_model(train)

_cat_encode = {"Gender":{"Male":1,"Female":0},"Customer Type":{"Loyal Customer":1,"disloyal Customer":0},
    "Type of Travel":{"Business travel":1,"Personal Travel":0},"Class":{"Business":2,"Eco Plus":1,"Eco":0}}
_enc = orig_clean[ORIGINAL_FEATURES].copy()
for col, m in _cat_encode.items(): _enc[col] = orig_clean[col].map(m)
orig_mean_maps = {}
for cols in [("Type of Travel",),("Class",),("Customer Type",),("Type of Travel","Class","Customer Type")]:
    key = _enc[list(cols)].astype(str).apply("|".join, axis=1)
    orig_mean_maps[cols] = pd.DataFrame({"key":key,"label":orig_y}).groupby("key")["label"].mean()
orig_prior = float(orig_y.mean())

def base_preprocess(df):
    df = df.copy(); df.drop(columns=["id"], inplace=True, errors="ignore")
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])
    cat_map = {"Gender":{"Male":1,"Female":0},"Customer Type":{"Loyal Customer":1,"disloyal Customer":0},
        "Type of Travel":{"Business travel":1,"Personal Travel":0},"Class":{"Business":2,"Eco Plus":1,"Eco":0}}
    for col, m in cat_map.items(): df[col] = df[col].map(m)
    df["total_delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    df["is_delayed"] = (df["Departure Delay in Minutes"] > 0).astype(int)
    df["avg_rating"] = df[RATING_COLS].mean(axis=1); df["min_rating"] = df[RATING_COLS].min(axis=1)
    df["max_rating"] = df[RATING_COLS].max(axis=1); df["rating_std"] = df[RATING_COLS].std(axis=1)
    df["zero_count"] = (df[RATING_COLS] == 0).sum(axis=1); df["has_zero"] = (df["zero_count"] > 0).astype(int)
    df["age_group"] = pd.cut(df["Age"], bins=[0,18,30,45,60,100], labels=[0,1,2,3,4]).astype(float)
    df["biz_biz"] = ((df["Type of Travel"]==1)&(df["Class"]==2)).astype(int)
    df["distance_bin"] = pd.cut(df["Flight Distance"], bins=[0,500,1500,float("inf")], labels=[0,1,2]).astype(float)
    df["weighted_rating"] = df["Online boarding"]*0.25 + df["Inflight entertainment"]*0.20 + df["Seat comfort"]*0.15 + df[["Inflight wifi service","On-board service","Leg room service"]].mean(axis=1)*0.40
    return df

train_p = base_preprocess(train)
FEATURE_COLS = [c for c in train_p.columns if c != "satisfaction"]
X_base = train_p[FEATURE_COLS].reset_index(drop=True)
y = train_p["satisfaction"].reset_index(drop=True)

KEY_PAIRS = [("Class","Type of Travel"),("Customer Type","Type of Travel"),("Class","Customer Type"),
    ("Gender","Class"),("Inflight wifi service","Online boarding"),("Inflight wifi service","Type of Travel"),
    ("Online boarding","Type of Travel"),("Seat comfort","Inflight entertainment"),
    ("Class","Inflight wifi service"),("Class","Online boarding")]
BASE_ENC_COLS = ["Gender","Customer Type","Type of Travel","Class","Inflight wifi service",
    "Ease of Online booking","Online boarding","Seat comfort","Inflight entertainment",
    "On-board service","Age","Flight Distance"]

def cat_key(df, cols):
    vals = df[list(cols)].astype(str); key = vals.iloc[:,0]
    for c in vals.columns[1:]: key = key.str.cat(vals[c], sep="|")
    return key

def add_distance_profile(X_tr, X_val):
    dist_col = "Flight Distance"; measure = RATING_COLS + ["Age","Departure Delay in Minutes","Arrival Delay in Minutes"]
    catp = ["Gender","Customer Type","Type of Travel","Class"]
    grouped = X_tr.groupby(dist_col)[measure].agg(["mean","std"]); counts = X_tr[dist_col].value_counts()
    proportions = {col: pd.crosstab(X_tr[dist_col], X_tr[col], normalize="index") for col in catp}
    defaults = {col: {"mean": X_tr[col].mean(), "std": X_tr[col].std()} for col in measure}
    results = []
    for frame in [X_tr, X_val]:
        feats = {"profile_route_count": frame[dist_col].map(counts).fillna(0).astype("float32")}
        for col in measure:
            fn = col.lower().replace(" ","_").replace("/","_").replace("-","_")
            mu = frame[dist_col].map(grouped[(col,"mean")]).fillna(defaults[col]["mean"]).astype("float32")
            sig = frame[dist_col].map(grouped[(col,"std")]).fillna(defaults[col]["std"]).astype("float32")
            feats[f"profile_{fn}_mean"] = mu; feats[f"profile_{fn}_std"] = sig
            feats[f"profile_{fn}_residual"] = (frame[col] - mu).astype("float32")
        for col, table in proportions.items():
            prior = X_tr[col].value_counts(normalize=True); fn = col.lower().replace(" ","_").replace("/","_").replace("-","_")
            for cat in table.columns:
                feats[f"profile_{fn}_{cat}_share"] = frame[dist_col].map(table[cat]).fillna(float(prior.get(cat,0.0))).astype("float32")
        results.append(pd.concat([frame, pd.DataFrame(feats, index=frame.index)], axis=1))
    return results

def add_frequency(X_tr, X_val):
    specs = [(c,) for c in BASE_ENC_COLS] + KEY_PAIRS; results = [X_tr.copy(), X_val.copy()]
    for cols in specs:
        name = "__".join(c.lower().replace(" ","_").replace("/","_") for c in cols)
        fm = cat_key(X_tr, cols).value_counts(normalize=True)
        for i, frame in enumerate([X_tr, X_val]):
            freq = cat_key(frame, cols).map(fm).fillna(0).astype("float32")
            results[i][f"freq_{name}"] = freq
            results[i][f"rarity_{name}"] = -np.log(freq.clip(lower=1.0/len(X_tr))).astype("float32")
    return results

def add_orig_means(X_tr, X_val):
    results = [X_tr.copy(), X_val.copy()]
    for cols, mapping in orig_mean_maps.items():
        feat = "orig_mean_" + "__".join(c.lower().replace(" ","_") for c in cols)
        for i, frame in enumerate([X_tr, X_val]):
            key = frame[list(cols)].astype(str).apply("|".join, axis=1)
            results[i][feat] = key.map(mapping).fillna(orig_prior).astype("float32")
    return results

def add_te(X_tr, X_val, y_tr, fold):
    def build_keys(df):
        keys = {}
        for col in BASE_ENC_COLS: keys[col] = df[col]
        for cols in KEY_PAIRS:
            name = "pair_" + "__".join(c.lower().replace(" ","_") for c in cols)
            keys[name] = df[cols[0]]*10 + df[cols[1]] if all(c in RATING_COLS for c in cols) else cat_key(df, cols).astype("category")
        keys["distance_250_bin"] = (df["Flight Distance"]//250).astype("float32"); keys["age_5_bin"] = (df["Age"]//5).astype("float32")
        return pd.DataFrame(keys, index=df.index)
    enc = TargetEncoder(target_type="binary", smooth=20.0, cv=5, shuffle=True, random_state=1042+fold)
    tr = enc.fit_transform(build_keys(X_tr), y_tr); va = enc.transform(build_keys(X_val))
    names = [f"te_{c}" for c in build_keys(X_tr).columns]
    return (pd.concat([X_tr, pd.DataFrame(tr.astype("float32"), columns=names, index=X_tr.index)], axis=1),
            pd.concat([X_val, pd.DataFrame(va.astype("float32"), columns=names, index=X_val.index)], axis=1))

lgb_params = dict(objective="binary", metric="auc", n_estimators=4000, learning_rate=0.04, num_leaves=63, max_depth=6,
    subsample=0.9, subsample_freq=1, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=5.0,
    min_child_samples=20, n_jobs=-1, verbosity=-1)
xgb_params = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", n_estimators=4000, learning_rate=0.04,
    max_depth=6, min_child_weight=5, subsample=0.9, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=5.0,
    max_bin=256, early_stopping_rounds=150, n_jobs=-1, verbosity=0)
SEEDS = [42, 7, 2024]

# 外层 fold 0: ia = folds 1-9
outer = list(StratifiedKFold(10, shuffle=True, random_state=42).split(X_base, y))
_, iv = outer[1]
ia = np.setdiff1d(np.arange(len(X_base)), iv)

teacher_oof = np.full(len(X_base), np.nan)
inner = StratifiedKFold(5, shuffle=True, random_state=42)
for fold, (tr, va) in enumerate(inner.split(ia, y[ia].to_numpy())):
    itr, iva = ia[tr], ia[va]
    X_tr, X_val = add_distance_profile(X_base.iloc[itr], X_base.iloc[iva])
    X_tr, X_val = add_frequency(X_tr, X_val)
    X_tr, X_val = add_orig_means(X_tr, X_val)
    X_tr, X_val = add_te(X_tr, X_val, y.iloc[itr], fold)
    for f in [X_tr, X_val]:
        f.replace([np.inf,-np.inf], 0, inplace=True); f.fillna(0, inplace=True)
        f = f.loc[:, ~f.columns.duplicated()].astype("float32")
    X_tr = X_tr[[c for c in X_tr.columns if X_tr[c].nunique()>1]]; X_val = X_val[X_tr.columns]
    oof_l = np.zeros(len(iva)); oof_x = np.zeros(len(iva))
    for s in SEEDS:
        ml = lgb.LGBMClassifier(**{**lgb_params, "random_state": s})
        ml.fit(X_tr, y.iloc[itr], eval_set=[(X_val, y.iloc[iva])], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof_l += ml.predict_proba(X_val)[:,1]/len(SEEDS)
        mx = xgb.XGBClassifier(**{**xgb_params, "random_state": s})
        mx.fit(X_tr, y.iloc[itr], eval_set=[(X_val, y.iloc[iva])], verbose=False)
        oof_x += mx.predict_proba(X_val)[:,1]/len(SEEDS)
    teacher_oof[iva] = 0.5*oof_l + 0.5*oof_x
    print(f"inner fold {fold}: route blend AUC={roc_auc_score(y.iloc[iva], teacher_oof[iva]):.5f}", flush=True)

np.save("cache/teacher_route_oof.npy", teacher_oof)
print(f"\nroute 腿教师 OOF 完成, 覆盖 {np.isfinite(teacher_oof).sum()} 行 (ia)", flush=True)
