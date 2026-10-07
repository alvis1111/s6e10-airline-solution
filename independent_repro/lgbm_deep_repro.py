"""深层 LightGBM 独立复现（fold 0）——busyaprime 公开源码（注明来源）。

= Busy 的 lgbm 腿：全 21 字段+10 交叉的 TE(auto 平滑)、LightGBM 原始数据教师、航线/频率/类别表示。
只跑 fold 0（10 折 seed 42 与自有融合对齐，SEED=42 复现方法），固定 10% 加入当前融合初筛。
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

DATA = "data/playground-series-s6e10"
REAL = r"C:\Users\GEM07\.cache\kagglehub\datasets\teejmahal20\airline-passenger-satisfaction\versions\1"
TGT = "satisfaction"
SEED = 42

CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
RATINGS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
           "Gate location", "Food and drink", "Online boarding", "Seat comfort",
           "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
           "Checkin service", "Cleanliness"]
NUMS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
BASE = CATS + NUMS + RATINGS
PAIRS = [("Class", "Type of Travel"), ("Customer Type", "Type of Travel"), ("Class", "Customer Type"),
         ("Gender", "Class"), ("Inflight wifi service", "Online boarding"),
         ("Inflight wifi service", "Type of Travel"), ("Online boarding", "Type of Travel"),
         ("Seat comfort", "Inflight entertainment"), ("Class", "Inflight wifi service"),
         ("Class", "Online boarding")]

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train[TGT].astype(int).to_numpy()


def is_texty(s):
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def load_original():
    o = pd.concat([pd.read_csv(f"{REAL}/train.csv"), pd.read_csv(f"{REAL}/test.csv")], ignore_index=True)
    o = o.drop(columns=[c for c in ("Unnamed: 0", "id", "Inflight service") if c in o.columns])
    o[TGT] = (o[TGT] == "satisfied").astype(int)
    return o


def build(tr, te, blocks):
    full = pd.concat([tr.drop(columns=[TGT]), te], ignore_index=True)
    X = full[BASE].copy()
    X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(0)
    if "org" in blocks or "origm" in blocks:
        o = load_original()
        o["Arrival Delay in Minutes"] = o["Arrival Delay in Minutes"].fillna(0)
    if "org" in blocks:
        g = o[TGT].mean()
        for c in BASE:
            st = o.groupby(c, observed=False)[TGT].mean()
            X[f"org_{c}"] = full[c].map(st).fillna(g).astype("float32").values
    if "origm" in blocks:
        Xo = o[BASE].copy()
        for c in CATS:
            cat = pd.CategoricalDtype(sorted(set(Xo[c].dropna()) | set(X[c].dropna())))
            Xo[c] = Xo[c].astype(cat)
        Xf = X[BASE].copy()
        for c in CATS:
            Xf[c] = Xf[c].astype(Xo[c].dtype)
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63,
                               colsample_bytree=0.6, subsample=0.8, subsample_freq=1,
                               reg_lambda=5.0, n_jobs=-1, random_state=SEED, verbose=-1)
        m.fit(Xo, o[TGT])
        X["orig_model"] = m.predict_proba(Xf)[:, 1].astype("float32")
    if "route" in blocks:
        fdk = full["Flight Distance"]
        coded = full[BASE].copy()
        coded["Arrival Delay in Minutes"] = coded["Arrival Delay in Minutes"].fillna(0)
        for c in CATS:
            coded[c] = coded[c].astype("category").cat.codes
        grp = coded.groupby(fdk)
        X["fd_cnt"] = fdk.map(fdk.value_counts()).astype("float32").values
        for c in BASE:
            if c == "Flight Distance":
                continue
            X[f"fdm_{c}"] = grp[c].transform("mean").astype("float32").values
            X[f"fdd_{c}"] = (coded[c] - X[f"fdm_{c}"]).astype("float32").values
    if "digit" in blocks:
        for c in NUMS:
            v = X[c].fillna(0)
            for k in range(-4, 4):
                X[f"{c}_d{k}"] = (v // (10.0 ** k) % 10).astype("int8")
    if "freq" in blocks:
        for c in list(X.columns):
            if c.startswith(("org_", "orig_model")):
                continue
            f = X[c].map(X[c].value_counts(normalize=True)).astype("float32")
            X[f"freq_{c}"] = f
            X[f"rar_{c}"] = (-np.log(f.clip(lower=1 / len(X)))).astype("float32")
    cat_cols = [c for c in X.columns if c in CATS or is_texty(X[c])]
    for c in cat_cols:
        X[c] = X[c].astype(str).astype("category")
    n = len(tr)
    return X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True), cat_cols


def add_te(Xa, ya, Xv, Xt, cols, k):
    enc = TargetEncoder(target_type="binary", smooth="auto", cv=5, random_state=SEED + k)
    A = Xa[cols].astype(str); V = Xv[cols].astype(str); T = Xt[cols].astype(str)
    ta = enc.fit_transform(A, ya); tv = enc.transform(V); tt = enc.transform(T)
    names = [f"te_{c}" for c in cols]
    return (pd.concat([Xa.reset_index(drop=True), pd.DataFrame(ta, columns=names)], axis=1),
            pd.concat([Xv.reset_index(drop=True), pd.DataFrame(tv, columns=names)], axis=1),
            pd.concat([Xt.reset_index(drop=True), pd.DataFrame(tt, columns=names)], axis=1))


def add_te2(Xa, ya, Xv, Xt, k):
    def pairs(F):
        return pd.DataFrame({f"{a}|{b}": F[a].astype(str) + "_" + F[b].astype(str) for a, b in PAIRS})
    enc = TargetEncoder(target_type="binary", smooth="auto", cv=5, random_state=SEED + 100 + k)
    Pa, Pv, Pt = pairs(Xa), pairs(Xv), pairs(Xt)
    ta = enc.fit_transform(Pa, ya); tv = enc.transform(Pv); tt = enc.transform(Pt)
    nm = [f"te2_{c}" for c in Pa.columns]
    return (pd.concat([Xa.reset_index(drop=True), pd.DataFrame(ta, columns=nm)], axis=1),
            pd.concat([Xv.reset_index(drop=True), pd.DataFrame(tv, columns=nm)], axis=1),
            pd.concat([Xt.reset_index(drop=True), pd.DataFrame(tt, columns=nm)], axis=1))


t0 = time.time()
X, Xt, cats = build(train, test, ["org", "origm", "digit", "freq", "route"])
print(f"特征 {X.shape[1]} 列（{time.time()-t0:.0f}s）", flush=True)

import sys
FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 0
folds = list(StratifiedKFold(10, shuffle=True, random_state=42).split(X, y))
ia, iv = folds[FOLD]
print(f"fold {FOLD}: train {len(ia)} / val {len(iv)}", flush=True)

Xa, Xv, Xtk = add_te(X.iloc[ia], y[ia], X.iloc[iv], Xt, BASE, FOLD)
Xa, Xv, Xtk = add_te2(Xa, y[ia], Xv, Xtk, FOLD)
print(f"加 TE 后 {Xa.shape[1]} 列", flush=True)

t1 = time.time()
m = lgb.LGBMClassifier(n_estimators=8000, learning_rate=0.02, max_depth=8, num_leaves=127,
                       colsample_bytree=0.5, subsample=0.9, subsample_freq=1, min_child_samples=20,
                       reg_lambda=5.0, reg_alpha=0.05, n_jobs=-1, random_state=SEED + FOLD, verbose=-1)
m.fit(Xa, y[ia], eval_set=[(Xv, y[iv])], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
pred = m.predict_proba(Xv)[:, 1]
print(f"fold {FOLD} 深层LGBM 单模型 AUC = {roc_auc_score(y[iv], pred):.6f}  ({time.time()-t1:.0f}s)", flush=True)
np.savez(f"cache/lgbm_deep_fold{FOLD}.npz", val_id=train["id"].iloc[iv].to_numpy(), pred=pred, y_val=y[iv])

# ---- 10% 固定加入当前融合 ----
route_aux = 0.5 * np.load("cache/oof_route_aux_l.npy") + 0.5 * np.load("cache/oof_route_aux_x.npy")
z = np.load("digit_confirm_results/refined_components.npz", allow_pickle=False)
order = {tid: i for i, tid in enumerate(train["id"].to_numpy())}
ctx = np.full(len(train), np.nan); ctx[[order[t] for t in z["train_id"]]] = z["context_oof"]
tabm = np.full(len(train), np.nan); tabm[[order[t] for t in z["train_id"]]] = z["route_tabm_oof"]
rk = lambda a: rankdata(a) / len(a)
fusion = 0.3 * rk(route_aux) + 0.3 * rk(ctx) + 0.4 * rk(tabm)
base = fusion[iv]
new = 0.9 * rk(base) + 0.1 * rk(pred)
print(f"原融合(fold {FOLD} val) = {roc_auc_score(y[iv], base):.6f}", flush=True)
print(f"+10% 深层LGBM        = {roc_auc_score(y[iv], new):.6f}  Δ {roc_auc_score(y[iv], new) - roc_auc_score(y[iv], base):+.6f}", flush=True)
