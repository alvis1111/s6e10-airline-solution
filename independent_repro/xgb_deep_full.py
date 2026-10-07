"""深层 XGBoost 完整 10 折（内层选轮数 + 外层重拟合）+ 10% 加入 0.96125 融合提交。

= xgb_deep_repro.py 的 10 折版。逐折 checkpoint（cache/partial/xgb_deep_full_f*.npz）。
基线 = 当前 0.96125 融合（Codex 的 oof_candidate/pred_candidate）。
"""
import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

DATA = "data/playground-series-s6e10"
REAL = r"C:\Users\GEM07\.cache\kagglehub\datasets\teejmahal20\airline-passenger-satisfaction\versions\1"
TGT = "satisfaction"
SEED = 42
NFOLD = 10

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
sub = pd.read_csv(f"{DATA}/sample_submission.csv")
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


def xgb_params(n_est, seed):
    return dict(n_estimators=n_est, learning_rate=0.02, max_depth=8, subsample=0.9,
                colsample_bytree=0.5, min_child_weight=5, reg_lambda=5.0, reg_alpha=0.05,
                tree_method="hist", device="cuda", enable_categorical=True, max_cat_to_onehot=8,
                random_state=seed)


t0 = time.time()
X, Xt, cats = build(train, test, ["org", "origm", "digit", "freq", "route"])
print(f"特征 {X.shape[1]} 列（{time.time()-t0:.0f}s）", flush=True)

os.makedirs("cache/partial", exist_ok=True)
folds = list(StratifiedKFold(NFOLD, shuffle=True, random_state=42).split(X, y))
oof = np.zeros(len(train), dtype=np.float64)
test_pred = np.zeros(len(test), dtype=np.float64)
t_start = time.time()

for fold, (ia, iv) in enumerate(folds):
    part = f"cache/partial/xgb_deep_full_f{fold}.npz"
    if os.path.exists(part):
        d = np.load(part)
        oof[iv] = d["oof"]; test_pred += d["test"] / NFOLD
        print(f"fold {fold} restored  auc {roc_auc_score(y[iv], oof[iv]):.6f}", flush=True)
        continue
    t1 = time.time()
    Xa, Xv, Xtk = add_te(X.iloc[ia], y[ia], X.iloc[iv], Xt, BASE, fold)
    Xa, Xv, Xtk = add_te2(Xa, y[ia], Xv, Xtk, fold)
    inner = StratifiedKFold(5, shuffle=True, random_state=0)
    i_tr, i_va = next(iter(inner.split(Xa, y[ia])))
    m_in = xgb.XGBClassifier(**xgb_params(12000, SEED + fold), eval_metric="auc", early_stopping_rounds=300)
    m_in.fit(Xa.iloc[i_tr], y[ia][i_tr], eval_set=[(Xa.iloc[i_va], y[ia][i_va])], verbose=False)
    best_iter = int(m_in.best_iteration or 12000)
    m_out = xgb.XGBClassifier(**xgb_params(best_iter, SEED + fold))
    m_out.fit(Xa, y[ia], verbose=False)
    oof[iv] = m_out.predict_proba(Xv)[:, 1]
    pt = m_out.predict_proba(Xtk)[:, 1]
    test_pred += pt / NFOLD
    np.savez(part, oof=oof[iv], test=pt)
    print(f"fold {fold}: it={best_iter}  auc {roc_auc_score(y[iv], oof[iv]):.6f}  ({time.time()-t1:.0f}s)", flush=True)

print(f"\n深层XGB 10折 OOF AUC = {roc_auc_score(y, oof):.6f}  (总 {time.time()-t_start:.0f}s)", flush=True)
np.save("cache/oof_xgb_deep.npy", oof)
np.save("cache/pred_xgb_deep.npy", test_pred)

# ---- 10% 加入 0.96125 融合提交 ----
base_z = np.load(r"C:/Users/GEM07/Documents/Codex/2026-10-02/https-www-kaggle-com-competitions-playground/outputs/lean_aux_confirmation/full_ed71a6690aa6/predictions.npz", allow_pickle=False)
tr_ord = {tid: i for i, tid in enumerate(train["id"].to_numpy())}
te_ord = {tid: i for i, tid in enumerate(test["id"].to_numpy())}
oof_cand = np.full(len(train), np.nan); oof_cand[[tr_ord[t] for t in base_z["train_id"]]] = base_z["oof_candidate"]
pred_cand = np.full(len(test), np.nan); pred_cand[[te_ord[t] for t in base_z["test_id"]]] = base_z["pred_candidate"]
assert not np.isnan(oof_cand).any() and not np.isnan(pred_cand).any()
rk = lambda a: rankdata(a) / len(a)
print(f"0.96125 融合 OOF = {roc_auc_score(y, oof_cand):.6f}", flush=True)

blend_oof = 0.9 * rk(oof_cand) + 0.1 * rk(oof)
blend_te = 0.9 * rk(pred_cand) + 0.1 * rk(test_pred)
print(f"+10% 深层XGB OOF = {roc_auc_score(y, blend_oof):.6f}  Δ {roc_auc_score(y, blend_oof) - roc_auc_score(y, oof_cand):+.6f}", flush=True)

sub["satisfaction"] = blend_te
sub.to_csv("submission_xgb_deep.csv", index=False)
print("saved -> submission_xgb_deep.csv", flush=True)
