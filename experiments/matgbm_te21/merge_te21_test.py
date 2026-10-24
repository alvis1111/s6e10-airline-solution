"""merge_te21_test — 把 TE21(全21列 smooth auto)+ 缓存 teacher 当普通特征并入 route_profile 强模型, 单折 A/B。

A = route_profile 现有特征管线(base_preprocess + orig_proba/orig_logit + 距离画像 +
    频率 + orig均值 + fold-safe TE smooth=20)
B = A + TE21(全21列 sklearn smooth='auto' OOF) + 缓存 teacher

单折(fold 0, 10-fold 同 route_profile), 单 seed, LGBM 同参数。
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

DATA = r"C:\Users\GEM07\kaggle\s6e10-airline\data\playground-series-s6e10"
REAL = r"C:\Users\GEM07\.cache\kagglehub\datasets\teejmahal20\airline-passenger-satisfaction\versions\1"
TEACHER = r"C:\Users\GEM07\Documents\Codex\2026-10-02\https-www-kaggle-com-competitions-playground\outputs\route_seed_experiment\production\teacher_predictions.npz"

RATING_COLS = ["Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
    "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness"]
CAT_COLS_RAW = ["Gender", "Customer Type", "Type of Travel", "Class"]
ORIGINAL_FEATURES = [c for c in ["Age","Flight Distance"]+RATING_COLS+["Departure Delay in Minutes","Arrival Delay in Minutes"]+CAT_COLS_RAW]

def row_hashes(df):
    return pd.util.hash_pandas_object(df[ORIGINAL_FEATURES].astype(str), index=False)

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

def add_distance_profile(X_tr, X_val):
    dist_col = "Flight Distance"
    measure = RATING_COLS + ["Age","Departure Delay in Minutes","Arrival Delay in Minutes"]
    catp = ["Gender","Customer Type","Type of Travel","Class"]
    grouped = X_tr.groupby(dist_col)[measure].agg(["mean","std"])
    counts = X_tr[dist_col].value_counts()
    proportions = {col: pd.crosstab(X_tr[dist_col], X_tr[col], normalize="index") for col in catp}
    defaults = {col: {"mean": X_tr[col].mean(), "std": X_tr[col].std()} for col in measure}
    results = []
    for frame in [X_tr, X_val]:
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

def add_frequency(X_tr, X_val):
    specs = [(c,) for c in BASE_ENC_COLS] + KEY_PAIRS
    results = [X_tr.copy(), X_val.copy()]
    for cols in specs:
        name = "__".join(c.lower().replace(" ","_").replace("/","_") for c in cols)
        fm = cat_key(X_tr, cols).value_counts(normalize=True)
        for i, frame in enumerate([X_tr, X_val]):
            freq = cat_key(frame, cols).map(fm).fillna(0).astype("float32")
            results[i][f"freq_{name}"] = freq
            results[i][f"rarity_{name}"] = -np.log(freq.clip(lower=1.0/len(X_tr))).astype("float32")
    return results

def add_orig_means(X_tr, X_val, orig_mean_maps, orig_prior):
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
        keys["distance_250_bin"] = (df["Flight Distance"]//250).astype("float32")
        keys["age_5_bin"] = (df["Age"]//5).astype("float32")
        return pd.DataFrame(keys, index=df.index)
    enc = TargetEncoder(target_type="binary", smooth=20.0, cv=5, shuffle=True, random_state=1042+fold)
    tr = enc.fit_transform(build_keys(X_tr), y_tr)
    va = enc.transform(build_keys(X_val))
    names = [f"te_{c}" for c in build_keys(X_tr).columns]
    return (pd.concat([X_tr, pd.DataFrame(tr.astype("float32"), columns=names, index=X_tr.index)], axis=1),
            pd.concat([X_val, pd.DataFrame(va.astype("float32"), columns=names, index=X_val.index)], axis=1))

def main():
    t0 = time.time()
    train = pd.read_csv(f"{DATA}/train.csv")
    test = pd.read_csv(f"{DATA}/test.csv")
    y = train['satisfaction'].to_numpy(np.int64)
    teacher = np.load(TEACHER)
    t_train = teacher['train'].astype(np.float32)
    print(f"数据加载 ({time.time()-t0:.1f}s)", flush=True)

    # original 数据 teacher (orig_proba/orig_logit), 复刻 route_profile
    orig_tr = pd.read_csv(f"{REAL}/train.csv")
    orig_te = pd.read_csv(f"{REAL}/test.csv")
    original = pd.concat([orig_tr, orig_te], ignore_index=True)
    original["satisfaction"] = (original["satisfaction"] == "satisfied").astype(int)
    original = original.drop(columns=["Unnamed: 0", "id", "Inflight service"], errors="ignore")
    comp_hashes = pd.concat([row_hashes(train), row_hashes(test)], ignore_index=True)
    orig_hashes = row_hashes(original)
    overlap = orig_hashes.isin(comp_hashes)
    orig_clean = original[~overlap].drop_duplicates().reset_index(drop=True)
    orig_y = orig_clean["satisfaction"].astype(int)
    orig_X = orig_clean[ORIGINAL_FEATURES].copy()
    _cat_vocab = {}
    for col in CAT_COLS_RAW:
        cats = sorted(orig_X[col].dropna().unique().tolist())
        _cat_vocab[col] = cats
        orig_X[col] = pd.Categorical(orig_X[col], categories=cats)
    for col in ORIGINAL_FEATURES:
        if col not in CAT_COLS_RAW: orig_X[col] = orig_X[col].astype("float32")
    orig_model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=600, learning_rate=0.05,
        max_depth=8, subsample=0.8, colsample_bytree=0.5, tree_method="hist",
        enable_categorical=True, n_jobs=-1, random_state=0, verbosity=0)
    orig_model.fit(orig_X, orig_y)
    def apply_orig_model(df):
        Xr = df[ORIGINAL_FEATURES].copy()
        for col in CAT_COLS_RAW: Xr[col] = pd.Categorical(Xr[col], categories=_cat_vocab[col])
        for col in ORIGINAL_FEATURES:
            if col not in CAT_COLS_RAW: Xr[col] = Xr[col].astype("float32")
        p = orig_model.predict_proba(Xr)[:, 1].astype("float32")
        logit = np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).astype("float32")
        return p, logit
    train["orig_proba"], train["orig_logit"] = apply_orig_model(train)
    print(f"orig_model 完成 ({time.time()-t0:.1f}s)", flush=True)

    # orig means
    _cat_encode = {"Gender": {"Male":1,"Female":0}, "Customer Type": {"Loyal Customer":1,"disloyal Customer":0},
        "Type of Travel": {"Business travel":1,"Personal Travel":0}, "Class": {"Business":2,"Eco Plus":1,"Eco":0}}
    _enc = orig_clean[ORIGINAL_FEATURES].copy()
    for col, m in _cat_encode.items(): _enc[col] = orig_clean[col].map(m)
    orig_mean_maps = {}
    for cols in [("Type of Travel",),("Class",),("Customer Type",),("Type of Travel","Class","Customer Type")]:
        key = _enc[list(cols)].astype(str).apply("|".join, axis=1)
        orig_mean_maps[cols] = pd.DataFrame({"key":key,"label":orig_y}).groupby("key")["label"].mean()
    orig_prior = float(orig_y.mean())

    train_p = base_preprocess(train)
    X_base = train_p[[c for c in train_p.columns if c != "satisfaction"]].reset_index(drop=True)
    y = train_p["satisfaction"].reset_index(drop=True)
    print(f"base_preprocess 完成 ({time.time()-t0:.1f}s)", flush=True)

    # TE21 (全21列 smooth auto, 嵌套 OOF)
    Z = train.drop(columns=['id','satisfaction']).astype(str)
    TE21_COLS = [f'te21_{i}' for i in range(Z.shape[1])]
    print(f"TE21 生成中 ({time.time()-t0:.1f}s)", flush=True)

    skf = StratifiedKFold(10, shuffle=True, random_state=42)
    tr, va = next(iter(skf.split(X_base, y)))
    enc21 = TargetEncoder(target_type='binary', smooth='auto', cv=5, shuffle=True, random_state=20261002)
    te21_oof = np.zeros((len(X_base), Z.shape[1]), dtype=np.float32)
    te21_oof[tr] = enc21.fit_transform(Z.iloc[tr], y.iloc[tr])
    te21_oof[va] = enc21.transform(Z.iloc[va])
    print(f"TE21 完成 ({time.time()-t0:.1f}s)", flush=True)

    # fold 0 特征管线
    X_tr, X_val = add_distance_profile(X_base.iloc[tr], X_base.iloc[va])
    X_tr, X_val = add_frequency(X_tr, X_val)
    X_tr, X_val = add_orig_means(X_tr, X_val, orig_mean_maps, orig_prior)
    X_tr, X_val = add_te(X_tr, X_val, y.iloc[tr], 0)
    for f in [X_tr, X_val]:
        f.replace([np.inf,-np.inf], 0, inplace=True); f.fillna(0, inplace=True)
        f = f.loc[:, ~f.columns.duplicated()].astype("float32")
    X_tr = X_tr[[c for c in X_tr.columns if X_tr[c].nunique()>1]]
    X_val = X_val[X_tr.columns]
    print(f"fold0 特征: {X_tr.shape[1]} 列 ({time.time()-t0:.1f}s)", flush=True)

    # 新增 TE21 + teacher
    te21_tr = pd.DataFrame(te21_oof[tr], columns=TE21_COLS, index=X_tr.index)
    te21_va = pd.DataFrame(te21_oof[va], columns=TE21_COLS, index=X_val.index)
    t_tr = pd.DataFrame({'teacher': t_train[tr]}, index=X_tr.index)
    t_va = pd.DataFrame({'teacher': t_train[va]}, index=X_val.index)
    X_tr_B = X_tr.copy()
    X_val_B = X_val.copy()
    X_tr_new = pd.concat([X_tr, te21_tr, t_tr], axis=1)
    X_val_new = pd.concat([X_val, te21_va, t_va], axis=1)

    lgb_params = dict(objective="binary", metric="auc", n_estimators=4000, learning_rate=0.04,
        num_leaves=63, max_depth=6, subsample=0.9, subsample_freq=1, colsample_bytree=0.85,
        reg_alpha=0.05, reg_lambda=5.0, min_child_samples=20, n_jobs=-1, verbosity=-1, random_state=42)

    def fit_eval(Xtr, Xva, ytr, yva):
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(150, verbose=False)])
        return roc_auc_score(yva, m.predict_proba(Xva)[:,1])

    t = time.time()
    a_A = fit_eval(X_tr_B, X_val_B, y.iloc[tr], y.iloc[va])
    print(f"[A] 现有管线 fold0 OOF = {a_A:.5f}  ({time.time()-t:.0f}s)", flush=True)
    t = time.time()
    a_B = fit_eval(X_tr_new, X_val_new, y.iloc[tr], y.iloc[va])
    print(f"[B] +TE21+teacher  fold0 OOF = {a_B:.5f}  ({time.time()-t:.0f}s)", flush=True)
    print(f"\nΔ(B-A) = {a_B-a_A:+.5f}", flush=True)
    print(f"总耗时 {time.time()-t0:.1f}s", flush=True)

if __name__ == '__main__':
    main()
