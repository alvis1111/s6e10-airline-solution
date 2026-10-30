"""S6E10 — 验证「原始数据逐行 KNN 邻居特征」是否补上最后 0.0007 的 gap。

思路：合成数据是从真实 129k 数据集加噪生成的，每个竞赛行在原始数据里都有一个
「最近的亲戚」，那个亲戚的真实标签是近乎免费的局部信号。我们现在只用了全局的
orig_proba/orig_logit（原始数据上训的 XGB 当特征），没用逐行邻居信息。

本脚本：
  1. 对齐 schema、去重叠（同 route_profile.py），得到干净原始数据 ~129.9k 行；
  2. 在标准化数值 + one-hot 类别的特征空间里做 KNN（k=20），给每个竞赛行加 5 列：
       knn_label / knn_label_w / knn_dist1 / knn_dist_mean / knn_label_std；
  3. 跑「route_profile 全套特征」5 折 LGBM（单种子，快速验证），同折训练
      「带 KNN」vs「不带 KNN」两个模型，报告 OOF 增量。

注意：本脚本是快速验证，用 5 折 + 单种子 + 只 LGBM（完整版是 10 折 3 种子 LGBM+XGB）。
我们要的是「带/不带 KNN 的 OOF 增量」这个相对量，绝对 OOF 会比完整版略低。
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
import torch

DATA = "data/playground-series-s6e10"
REAL = r"C:\Users\GEM07\.cache\kagglehub\datasets\teejmahal20\airline-passenger-satisfaction\versions\1"

RATING_COLS = ["Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
    "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness"]
CAT_COLS_RAW = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
ORIGINAL_FEATURES = [c for c in
    ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
    + RATING_COLS + CAT_COLS_RAW]

K = 20
KNN_COLS = ["knn_label", "knn_label_w", "knn_dist1", "knn_dist_mean", "knn_label_std"]

# ---- 读数据 ----
train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train["satisfaction"].to_numpy(dtype=np.int64)

# ---- 原始数据：对齐 schema + 标签二值化 + 去重叠 ----
orig_tr = pd.read_csv(f"{REAL}/train.csv")
orig_te = pd.read_csv(f"{REAL}/test.csv")
original = pd.concat([orig_tr, orig_te], ignore_index=True)
original["satisfaction"] = (original["satisfaction"] == "satisfied").astype(int)
# 原始数据比竞赛多一列 "Inflight service"（竞赛把它去掉了），KNN 只能用共有列
original = original.drop(columns=["Unnamed: 0", "id", "Inflight service"], errors="ignore")

def row_hashes(df):
    return pd.util.hash_pandas_object(df[ORIGINAL_FEATURES].astype(str), index=False)

comp_hashes = pd.concat([row_hashes(train), row_hashes(test)], ignore_index=True)
orig_hashes = row_hashes(original)
overlap = orig_hashes.isin(comp_hashes)
orig_clean = original[~overlap].drop_duplicates().reset_index(drop=True)
orig_y = orig_clean["satisfaction"].astype(int)
print(f"原始数据: {len(original)} -> 去重叠后 {len(orig_clean)}", flush=True)

# ---- KNN 特征空间：标准化数值 + one-hot 类别 ----
CAT_MAP = {
    "Gender":         {"Male": 0, "Female": 1},
    "Customer Type":  {"Loyal Customer": 0, "disloyal Customer": 1},
    "Type of Travel": {"Business travel": 0, "Personal Travel": 1},
    "Class":          {"Business": 0, "Eco Plus": 1, "Eco": 2},
}

def numeric_matrix(df):
    num = df[NUM_COLS + RATING_COLS].copy().astype(float)
    num["Arrival Delay in Minutes"] = num["Arrival Delay in Minutes"].fillna(num["Departure Delay in Minutes"])
    return num.fillna(0).to_numpy()

def one_hot_cats(df):
    parts = []
    for c in CAT_COLS_RAW:
        v = df[c].map(CAT_MAP[c]).fillna(0).astype(int)
        n = max(CAT_MAP[c].values()) + 1
        for i in range(n):
            parts.append((v == i).astype(np.float32))
    return np.column_stack(parts)

ref_num = numeric_matrix(orig_clean)
scaler = StandardScaler().fit(ref_num)
ref = np.hstack([scaler.transform(ref_num), one_hot_cats(orig_clean)]).astype(np.float32)
ref_y = orig_y.to_numpy(dtype=np.int64)
print(f"KNN 参考矩阵: {ref.shape}, 维度 = {ref.shape[1]}", flush=True)

# GPU 暴力 KNN（torch）：ball_tree 在 26 维太慢（~2.3ms/行），GPU 分块精确计算快 100x
device = "cuda" if torch.cuda.is_available() else "cpu"
ref_t = torch.from_numpy(ref).to(device)
ref_y_t = torch.from_numpy(ref_y).to(device)
ref_norm = (ref_t * ref_t).sum(1)
print(f"KNN device = {device}, 参考行数 = {ref.shape[0]}", flush=True)

def knn_query(X, chunk=2048):
    """X: (M, D) float32 -> (dist (M,K), idx (M,K))。GPU 分块算欧氏距离 + topk。"""
    M = len(X)
    dist = np.zeros((M, K), dtype=np.float32)
    idx = np.zeros((M, K), dtype=np.int64)
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        q = torch.from_numpy(X[s:e]).to(device)
        sim = q @ ref_t.T                               # (c, N)
        d2 = (q * q).sum(1, keepdim=True) + ref_norm.unsqueeze(0) - 2.0 * sim
        d2 = torch.clamp(d2, min=0.0)
        d, i = torch.topk(d2, k=K, dim=1, largest=False)
        dist[s:e] = d.sqrt().cpu().numpy()
        idx[s:e] = i.cpu().numpy()
        del sim, d2, d, i
    return dist, idx

def knn_features(df):
    X = np.hstack([scaler.transform(numeric_matrix(df)), one_hot_cats(df)]).astype(np.float32)
    dist, idx = knn_query(X)
    lab = ref_y[idx]                       # (N, K) 邻居真实标签
    inv = 1.0 / (dist + 1e-6)
    w = inv / inv.sum(axis=1, keepdims=True)
    out = pd.DataFrame(index=df.index)
    out["knn_label"]      = lab.mean(axis=1).astype(np.float32)
    out["knn_label_w"]    = (lab * w).sum(axis=1).astype(np.float32)
    out["knn_dist1"]      = dist[:, 0]
    out["knn_dist_mean"]  = dist.mean(axis=1)
    out["knn_label_std"]  = lab.std(axis=1).astype(np.float32)
    return out

t0 = time.time()
knn_tr = knn_features(train)
knn_te = knn_features(test)
print(f"KNN 查询 train+test 用时 {time.time()-t0:.1f}s", flush=True)

# 诊断：最近邻距离分布（判断合成行离原始行有多近，距离极小≈可能接近复制）
for name, kf in [("train", knn_tr), ("test", knn_te)]:
    q = np.percentile(kf["knn_dist1"].values, [1, 25, 50, 75, 99])
    print(f"knn_dist1 分位数 [{name}] p1/p25/p50/p75/p99 = {np.round(q, 3).tolist()}", flush=True)

for name, df in [("train", train), ("test", test)]:
    kf = knn_tr if name == "train" else knn_te
    for c in KNN_COLS:
        df[c] = kf[c].values

# ---- 原始数据上训 XGB 当特征生成器（同 route_profile.py）----
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
    enable_categorical=True, n_jobs=-1, random_state=0, verbosity=0)
orig_model.fit(_orig_X, orig_y)

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

# 原始目标均值（同 route_profile.py）
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

# ---- base preprocess（同 route_profile.py）----
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

train_p = base_preprocess(train)
test_p = base_preprocess(test)
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
    vals = df[list(cols)].astype(str)
    key = vals.iloc[:, 0]
    for c in vals.columns[1:]:
        key = key.str.cat(vals[c], sep="|")
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
            feats[f"profile_{fn}_mean"] = mu
            feats[f"profile_{fn}_std"] = sig
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
        for col in BASE_ENC_COLS:
            keys[col] = df[col]
        for cols in KEY_PAIRS:
            name = "pair_" + "__".join(c.lower().replace(" ", "_") for c in cols)
            keys[name] = df[cols[0]] * 10 + df[cols[1]] if all(c in RATING_COLS for c in cols) else cat_key(df, cols).astype("category")
        keys["distance_250_bin"] = (df["Flight Distance"] // 250).astype("float32")
        keys["age_5_bin"] = (df["Age"] // 5).astype("float32")
        return pd.DataFrame(keys, index=df.index)
    enc = TargetEncoder(target_type="binary", smooth=20.0, cv=5, shuffle=True, random_state=1042 + fold)
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

# ---- 5 折 LGBM（单种子），同折训「带/不带 KNN」两个模型 ----
skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof_with = np.zeros(len(X_base))
oof_without = np.zeros(len(X_base))

for fold, (tr, va) in enumerate(skf.split(X_base, y)):
    t = time.time()
    X_tr, X_val, X_tst = add_distance_profile(X_base.iloc[tr], X_base.iloc[va], X_test_base)
    X_tr, X_val, X_tst = add_frequency(X_tr, X_val, X_tst)
    X_tr, X_val, X_tst = add_orig_means(X_tr, X_val, X_tst)
    X_tr, X_val, X_tst = add_te(X_tr, X_val, X_tst, y.iloc[tr], fold)
    for f in [X_tr, X_val, X_tst]:
        f.replace([np.inf, -np.inf], 0, inplace=True)
        f.fillna(0, inplace=True)
        f = f.loc[:, ~f.columns.duplicated()].astype("float32")
    X_tr = X_tr[[c for c in X_tr.columns if X_tr[c].nunique() > 1]]
    X_val = X_val[X_tr.columns]
    feats_all = X_tr.columns.tolist()
    feats_noknn = [c for c in feats_all if c not in KNN_COLS]
    if fold == 0:
        print(f"特征数: 带KNN={len(feats_all)}, 不带KNN={len(feats_noknn)}", flush=True)

    m_with = lgb.LGBMClassifier(**{**lgb_params, "random_state": 42})
    m_with.fit(X_tr[feats_all], y.iloc[tr], eval_set=[(X_val[feats_all], y.iloc[va])],
               callbacks=[lgb.early_stopping(150, verbose=False)])
    oof_with[va] = m_with.predict_proba(X_val[feats_all])[:, 1]

    m_without = lgb.LGBMClassifier(**{**lgb_params, "random_state": 42})
    m_without.fit(X_tr[feats_noknn], y.iloc[tr], eval_set=[(X_val[feats_noknn], y.iloc[va])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    oof_without[va] = m_without.predict_proba(X_val[feats_noknn])[:, 1]

    print(f"fold {fold}: 带KNN={roc_auc_score(y.iloc[va], oof_with[va]):.5f} "
          f"不带KNN={roc_auc_score(y.iloc[va], oof_without[va]):.5f}  ({time.time()-t:.0f}s)", flush=True)

auc_with = roc_auc_score(y, oof_with)
auc_without = roc_auc_score(y, oof_without)
print("\n" + "=" * 56)
print(f"  5折 LGBM(单种子)  OOF  不带KNN = {auc_without:.5f}")
print(f"  5折 LGBM(单种子)  OOF  带KNN   = {auc_with:.5f}")
print(f"  KNN 增量                       = {auc_with - auc_without:+.5f}")
print("=" * 56)

np.save("cache/oof_knn_with.npy", oof_with)
np.save("cache/oof_knn_without.npy", oof_without)
print("saved cache/oof_knn_with.npy / oof_knn_without.npy", flush=True)
