"""复现 ewtherka 融合器：显式删 gpx_stack245 后，flat LR C=0.003 vs family C=0.025 的 80/20 融合对比。

与 notebook 源码一致：18 源头加载 → 过滤(scale/长度/AUC/agreement/drift/近重复) → 删 gpx_stack245 → 247 成员
→ 同一套成员上跑 flat LR 与 family LR（各配 20% LGB），比 fold-mean AUC。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

DATA = "data/playground-series-s6e10"
M = "pulled_members"
G = "pulled_golem"

train = pd.read_csv(f"{DATA}/train.csv", usecols=["id", "satisfaction"])
train_ids = train["id"].to_numpy()
test_ids = pd.read_csv(f"{DATA}/test.csv", usecols=["id"])["id"].to_numpy()
y = train["satisfaction"].astype(int).to_numpy()

fold = np.zeros(len(y), int)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
    fold[va] = k
assert (np.load(f"{G}/folds_seed42.npy") == fold).all(), "folds mismatch"


def first(d, name):
    hits = sorted(glob.glob(os.path.join(d, "**", name), recursive=True), key=len)
    if not hits:
        raise FileNotFoundError(f"{name} in {d}")
    return hits[0]


def by_id(df, ids, col):
    s = df.set_index("id")[col]
    assert s.index.is_unique, "dup ids"
    s = s.reindex(ids)
    assert s.notna().all(), f"{int(s.isna().sum())} missing ids"
    return s.to_numpy(np.float64)


def npy_pairs(d, prefix, pattern="oof_*.npy"):
    for f in sorted(glob.glob(os.path.join(d, "**", pattern), recursive=True)):
        name = os.path.basename(f)[4:-4]
        yield prefix + name, np.load(f), np.load(os.path.join(os.path.dirname(f), f"test_{name}.npy")), "row order"


def npz_pairs(d, prefix, pattern="*.npz", strip=0):
    for f in sorted(glob.glob(os.path.join(d, "**", pattern), recursive=True)):
        with np.load(f) as z:
            if "oof" in z.files and "test" in z.files:
                yield prefix + os.path.basename(f)[strip:-4], z["oof"], z["test"], "row order"


SOURCES = []


def reg(key, d):
    def deco(fn):
        SOURCES.append((key, d, fn))
        return fn
    return deco


@reg("c124", f"{M}/c124")
def _c124(d): yield from npy_pairs(d, "")


@reg("tfm", f"{M}/tfm")
def _tfm(d): yield from npy_pairs(d, "tfm_")


@reg("golem", G)
def _golem(d):
    oof = pd.read_parquet(f"{d}/oof_wide.parquet")
    test = pd.read_parquet(f"{d}/test_wide.parquet")
    for c in [c for c in oof.columns if c not in ("id", "fold")]:
        yield "gl_" + c, oof[c].to_numpy(), test[c].to_numpy(), "row order"


@reg("abdullah", f"{M}/abdullah")
def _abd(d):
    for m in ("tabpfn_catfd10", "tabpfn_raw", "tabpfn_te_cnt", "cat_v3", "xgb_v3"):
        o = pd.read_csv(first(d, f"{m}_oof.csv"))
        t = pd.read_csv(first(d, f"{m}_test.csv"))
        yield "ab_" + m, by_id(o, train_ids, "pred"), by_id(t, test_ids, "pred"), "id"


@reg("sadam", f"{M}/sadam")
def _sadam(d):
    yield from npz_pairs(d, "st_", "preds_*.npz", strip=6)
    for n in ("orig_cat_pred", "orig_model_pred"):
        with np.load(first(d, f"{n}.npz")) as z:
            yield "st_" + n, z["train"], z["test"], "row order"


@reg("forge", f"{M}/forge")
def _forge(d): yield from npy_pairs(d, "am_", "oof_forge_*.npy")


@reg("sachith", f"{M}/sachith")
def _sachith(d): yield from npz_pairs(d, "s7_")


@reg("koumeimaki", f"{M}/koumeimaki")
def _km(d):
    assert (pd.read_csv(first(d, "train_ids_folds.csv"))["id"].to_numpy() == train_ids).all()
    assert (pd.read_csv(first(d, "test_ids.csv"))["id"].to_numpy() == test_ids).all()
    for name, o, t, _ in npy_pairs(d, "km_"):
        yield name, o, t, "id"


@reg("gp_tabpfn", f"{M}/gp_tabpfn")
def _gpt(d):
    for f in sorted(glob.glob(os.path.join(d, "**", "oof_*.csv"), recursive=True)):
        n = os.path.basename(f)[4:-4]
        yield "gpt_" + n, by_id(pd.read_csv(f), train_ids, "pred"), by_id(pd.read_csv(first(d, f"test_{n}.csv")), test_ids, "pred"), "id"


@reg("gp_extra", f"{M}/gp_extra")
def _gpx(d): yield from npy_pairs(d, "gpx_")


@reg("dmitriy", f"{M}/dmitriy")
def _dk(d): yield "dk_cb_cat", np.load(first(d, "oof_cb_cat.npy")), np.load(first(d, "test_cb_cat.npy")), "row order"


@reg("kagan_abl", f"{M}/kagan_abl")
def _kga(d): yield from npy_pairs(d, "kga_")


@reg("yekenot", f"{M}/yekenot")
def _yk(d):
    o = pd.read_csv(first(d, "oof_preds.csv"))
    t = pd.read_csv(first(d, "submission.csv"))
    yield "yk_realmlp", by_id(o, train_ids, "satisfaction"), by_id(t, test_ids, "satisfaction"), "id"


@reg("hermengardo", f"{M}/hermengardo")
def _hm(d): yield "hm_cat", np.load(first(d, "oof_train.npy")), np.load(first(d, "oof_test.npy")), "row order"


@reg("kratosyan", f"{M}/kratosyan")
def _kr(d): yield from npz_pairs(d, "kr_", "*_route_ogaux.npz")


@reg("kagan_enc", f"{M}/kagan_enc")
def _kge(d):
    o = pd.read_csv(first(d, "oof_members.csv"))
    t = pd.read_csv(first(d, "test_members.csv"))
    for c in [c for c in o.columns if c != "id"]:
        yield "kg_" + c, by_id(o, train_ids, c), by_id(t, test_ids, c), "id"


@reg("aman_v5", f"{M}/aman_v5")
def _av5(d):
    with np.load(first(d, "oof_artifacts.npz")) as z:
        assert (z["y"] == y).all()
        for m in ("lgbm_champion", "lgbm_te", "lgbm_te_rows", "xgb_gpu", "catboost_gpu"):
            yield "av5_" + m, z["oof_" + m], z["test_" + m], "row order"


@reg("kozy", f"{M}/kozy")
def _kz(d):
    yield "kz_source_transfer", np.load(first(d, "source_transfer_oof.npy")), np.load(first(d, "source_transfer_test.npy")), "row order"


# ---------------- load + filter ----------------
MIN_AUC, MAX_AUC, AGREE_GAP, DRIFT, DUP_RHO = 0.90, 0.9625, 0.03, 0.03, 0.9999


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-15, 1 - 1e-15)
    return np.log(p / (1 - p))


def is_rank(v):
    q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
    return bool(v.min() >= 0 and v.max() <= 1 and np.abs(q - [0.1, 0.25, 0.5, 0.75, 0.9]).max() < 0.01)


def corr_cols(Z, r):
    Zc, rc = Z - Z.mean(0), r - r.mean()
    return (Zc * rc[:, None]).sum(0) / np.sqrt((Zc ** 2).sum(0) * (rc ** 2).sum())


rows, OOFL, TSTL = [], [], []
for key, d, load in SOURCES:
    try:
        got = list(load(d))
    except Exception as e:
        rows.append(dict(source=key, member="(all)", status=f"skipped: {type(e).__name__}: {e}"[:120]))
        continue
    for name, o, t, how in got:
        o = np.asarray(o, np.float64).ravel()
        t = np.asarray(t, np.float64).ravel()
        r = dict(source=key, member=name, status="")
        if o.shape != y.shape or t.shape != test_ids.shape:
            r["status"] = f"wrong length {o.shape[0]:,}/{t.shape[0]:,}"
        elif not (np.isfinite(o).all() and np.isfinite(t).all()):
            r["status"] = "NaN or inf"
        else:
            already = bool(o.min() < 0 or o.max() > 1 or t.min() < 0 or t.max() > 1)
            p_o, p_t = (1 / (1 + np.exp(-o)), 1 / (1 + np.exp(-t))) if already else (o, t)
            r["drift"] = abs(p_t.mean() - p_o.mean())
            zo, zt = (o, t) if already else (logit(o), logit(t))
            OOFL.append(np.clip(zo, -30, 30).astype(np.float32))
            TSTL.append(np.clip(zt, -30, 30).astype(np.float32))
            r["col"] = len(OOFL) - 1
        rows.append(r)

CHK = pd.DataFrame(rows)
CHK["col"] = CHK["col"].fillna(-1).astype(int)
live = CHK["col"] >= 0
Zall = np.column_stack(OOFL)
Tall = np.column_stack(TSTL)
CHK.loc[live, "oof_auc"] = [roc_auc_score(y, Zall[:, c]) for c in CHK.loc[live, "col"]]

BASE = CHK[live & CHK["source"].isin(["c124", "tfm"])]
assert len(BASE) == 135, f"base pool = {len(BASE)}"
ref_o = np.median(Zall[:, BASE["col"].to_numpy()], 1)
ref_t = np.median(Tall[:, BASE["col"].to_numpy()], 1)
CHK.loc[live, "agree_oof"] = corr_cols(Zall[:, CHK.loc[live, "col"]], ref_o)
CHK.loc[live, "agree_test"] = corr_cols(Tall[:, CHK.loc[live, "col"]], ref_t)

new = live & ~CHK.index.isin(BASE.index) & (CHK["status"] == "")
for i in CHK.index[new]:
    r = CHK.loc[i]
    if r["oof_auc"] < MIN_AUC:
        CHK.loc[i, "status"] = f"OOF AUC {r['oof_auc']:.3f} too low"
    elif r["oof_auc"] > MAX_AUC:
        CHK.loc[i, "status"] = f"OOF AUC {r['oof_auc']:.4f} too high"
    elif r["agree_oof"] - r["agree_test"] > AGREE_GAP:
        CHK.loc[i, "status"] = "agreement gap"
    elif r["drift"] > DRIFT:
        CHK.loc[i, "status"] = "mean drift"

order = list(BASE["col"]) + [int(c) for c in CHK.loc[new & (CHK["status"] == ""), "col"]]
Rk = np.column_stack([rankdata(Zall[:, c]) for c in order])
Rk -= Rk.mean(0)
Rk /= np.sqrt((Rk ** 2).mean(0))
RHO = np.clip((Rk.T @ Rk) / len(Rk), -1, 1)
kept = list(range(len(BASE)))
for j in range(len(BASE), len(order)):
    best = max(kept, key=lambda m: RHO[j, m])
    i = CHK.index[CHK["col"] == order[j]][0]
    if RHO[j, best] > DUP_RHO:
        CHK.loc[i, "status"] = f"near-copy rho {RHO[j, best]:.6f}"
    else:
        kept.append(j)
FULL_COLS = [order[j] for j in kept]
NAMES = [CHK.loc[CHK["col"] == c, "member"].iloc[0] for c in FULL_COLS]
SRCS = [CHK.loc[CHK["col"] == c, "source"].iloc[0] for c in FULL_COLS]

# ---- 显式删 gpx_stack245 ----
before = len(FULL_COLS)
drops = [j for j, n in enumerate(NAMES) if n == "gpx_stack245"]
if drops:
    j0 = drops[0]
    FULL_COLS = FULL_COLS[:j0] + FULL_COLS[j0 + 1:]
    NAMES = NAMES[:j0] + NAMES[j0 + 1:]
    SRCS = SRCS[:j0] + SRCS[j0 + 1:]
print(f"过滤后成员 {before}，其中 gpx_stack245 索引 {drops}；删除后 = {len(FULL_COLS)}")
print(f"来源分布: {pd.Series(SRCS).value_counts().to_dict()}")

Z = Zall[:, FULL_COLS].astype(np.float64)
ZT = Tall[:, FULL_COLS].astype(np.float64)
NTR = len(y)


def family(n):
    if 'tabpfn' in n or n.startswith('pfn_'):
        return 'tabpfn'
    if n.startswith('tfm_') or 'tabicl' in n or 'tabm' in n:
        return 'tfm'
    if 'realmlp' in n or 'rmlp' in n or 'mlp' in n:
        return 'mlp'
    return 'gbdt'


FAM = np.array([family(n) for n in NAMES])
SCALE = np.array([{'tabpfn': 3.0, 'tfm': 1.5}.get(f, 1.0) for f in FAM])
print("family counts:", {f: int((FAM == f).sum()) for f in ('gbdt', 'mlp', 'tabpfn', 'tfm')})

GB_EXCLUDE = {'tfm', 'golem', 'sadam', 'koumeimaki', 'kagan_enc', 'aman_v5', 'forge', 'kozy'}
GB_COLS = [i for i, s in enumerate(SRCS) if s not in GB_EXCLUDE]
print(f"LR stacker {Z.shape[1]} members, LGB stacker {len(GB_COLS)} members")

P = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000, feature_fraction=0.5,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=4, seed=0)


def fit_gb(A, ya):
    ai = np.random.default_rng(0).permutation(len(A))
    cut = int(len(ai) * 0.9)
    return lgb.train(P, lgb.Dataset(A[ai[:cut]], ya[ai[:cut]]), 3000,
                     valid_sets=[lgb.Dataset(A[ai[cut:]], ya[ai[cut:]])],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def nested_gb():
    o = np.zeros(NTR)
    A = Z[:, GB_COLS]
    for k in range(5):
        a, b = fold != k, fold == k
        m = fit_gb(A[a], y[a])
        o[b] = m.predict(A[b], raw_score=True, num_iteration=m.best_iteration)
    return o


zs = lambda v: (v - v.mean()) / v.std()
fa = lambda o: np.array([roc_auc_score(y[fold == k], o[fold == k]) for k in range(5)])


def nested_lr(Zm, C):
    o = np.zeros(NTR)
    for k in range(5):
        a, b = fold != k, fold == k
        m = LogisticRegression(C=C, tol=1e-6, max_iter=20000).fit(Zm[a], y[a])
        o[b] = m.decision_function(Zm[b])
    return o


W = 0.2
X = {'flat': (Z, ZT), 'family': (Z * SCALE, ZT * SCALE)}

GB = nested_gb()
print(f"\nLGB stacker alone fold-mean = {fa(GB).mean():.6f}")

results = {}
for sch, C in [('flat', 0.003), ('family', 0.025)]:
    lr = nested_lr(X[sch][0], C)
    lr_only = fa(lr)
    blend = (1 - W) * zs(lr) + W * zs(GB)
    bl = fa(blend)
    results[sch] = (lr_only, bl)
    print(f"\n{sch:6s} C={C:<6g}: plain LR {lr_only.mean():.6f} | 80/20 blend {bl.mean():.6f}")
    print(f"  LR folds {[round(v, 6) for v in lr_only]}")
    print(f"  BL folds {[round(v, 6) for v in bl]}")

flat_bl = results['flat'][1]
fam_bl = results['family'][1]
d = fam_bl - flat_bl
se = d.std(ddof=1) / np.sqrt(5)
print(f"\n===== family vs flat (80/20 blend, 247 成员无 gpx_stack245) =====")
print(f"flat   fold-mean {flat_bl.mean():.6f}")
print(f"family fold-mean {fam_bl.mean():.6f}")
print(f"增益   {d.mean():+.6f}  (SE {se:.6f}, {(d > 0).sum()}/5 折正)")
print(f"per-fold Δ = {[round(v, 6) for v in d]}")
