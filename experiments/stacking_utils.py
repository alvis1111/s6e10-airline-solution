"""stacking_utils.py — 从 ewtherka L2 stack（S6E10）提炼的可复用 stacking 工具

四个组件（都可单独 import 复用）：
  1. logit 口径统一    —— to_logit / is_rank
  2. 成员 QC 四件套    —— qc_members：AUC 上限反泄漏 + agreement gap + mean drift + 近重复去重
  3. 分族缩放正则      —— classify_family / family_scale
  4. 诚实 CV 选型      —— make_folds / nested_lr_cv / fold_mean_auc / honest_select

设计原则（来自复盘教训）：
  - 选 stacking 方案永远用嵌套 CV + 跨种子，不碰公榜/OOF 搜权（搜权会高估）。
  - 成员进 stacker 前先做 QC，尤其「OOF AUC 太高」要当泄漏/过拟合砍掉。

依赖：numpy / pandas / scipy / sklearn。lightgbm 仅在 lgb_stacker_cv 里用到（懒加载）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

CLIP = 30.0


# ---------- 1. logit 口径统一 ----------

def is_rank(v, tol=0.01) -> bool:
    """判断向量是不是百分位 rank（0~1 且分位数贴合 [0.1,...,0.9]）。"""
    v = np.asarray(v, float)
    if not (v.min() >= 0.0 and v.max() <= 1.0):
        return False
    q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
    return bool(np.abs(q - np.array([0.1, 0.25, 0.5, 0.75, 0.9])).max() < tol)


def to_logit(v, clip=CLIP):
    """把概率/rank/logit 统一到裁剪 logit。

    超出 [0,1] 视为已是 logit（保留、仅 clip）；在 [0,1] 内视为概率做 logit。
    百分位 rank 也会被 logit（单调变换，无损）。
    """
    v = np.asarray(v, float)
    if v.min() < 0.0 or v.max() > 1.0:
        return np.clip(v, -clip, clip)
    p = np.clip(v, 1e-15, 1 - 1e-15)
    return np.clip(np.log(p / (1 - p)), -clip, clip)


def zscore(v):
    """标准化到 0 均值单位方差（用于 LR 与 LGB 的 80/20 融合）。"""
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std()


# ---------- 2. folds / 嵌套 CV ----------

def make_folds(y, n_splits=5, seed=42):
    """StratifiedKFold → [(tr, va), ...]（默认与 ewtherka 一致 seed 42）。"""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def fold_mean_auc(y, oof, folds):
    """按折算 AUC 再平均（ewtherka 的 fold-mean 口径）。"""
    return float(np.mean([roc_auc_score(y[va], oof[va]) for _, va in folds]))


def nested_lr_cv(X, y, folds, C=0.025):
    """嵌套 LR：每折在 tr 上 fit、在 va 上 decision_function，拼成 OOF。"""
    o = np.zeros(len(y), float)
    for tr, va in folds:
        m = LogisticRegression(C=C, tol=1e-6, max_iter=20000).fit(X[tr], y[tr])
        o[va] = m.decision_function(X[va])
    return o


# ---------- 3. 分族缩放正则 ----------

def classify_family(name) -> str:
    """按名字把成员归族（tabpfn / tfm / mlp / gbdt）。"""
    n = str(name).lower()
    if 'tabpfn' in n or n.startswith('pfn_') or 'tabicl' in n:
        return 'tabpfn'
    if 'tfm' in n or 'tabm' in n or 'tabr' in n or 'ftt' in n:
        return 'tfm'
    if 'realmlp' in n or 'rmlp' in n or 'mlp' in n:
        return 'mlp'
    return 'gbdt'


DEFAULT_FAMILY_SCALE = {'tabpfn': 3.0, 'tfm': 1.5}


def family_scale(names, classify=None, scale_map=DEFAULT_FAMILY_SCALE):
    """进 LR 前按族缩放特征。

    机制：LR 的 L2 下把特征放大 s 倍 ≈ 该族惩罚 ÷ s²；所以 tabpfn/tfm 放大 = 更信任基础模型。
    返回 (families, scales)。
    """
    fam = [classify(n) if classify else classify_family(n) for n in names]
    scales = np.array([scale_map.get(f, 1.0) for f in fam], float)
    return fam, scales


# ---------- 4. 成员 QC 四件套 ----------

def _corr_cols(Z, r):
    Zc = Z - Z.mean(0)
    rc = r - r.mean()
    return (Zc * rc[:, None]).sum(0) / np.sqrt((Zc ** 2).sum(0) * (rc ** 2).sum())


def qc_members(OOF, TEST, y, names=None, base_idx=None,
               min_auc=0.90, max_auc=0.9625,
               agree_gap=0.03, drift=0.03, dup_rho=0.9999):
    """成员 QC：返回 (keep_mask, report)。

    OOF / TEST：形状 (n, m) / (n_test, m)，建议已是 logit（内部仍会 clip）。
    y：(n,) 0/1。names：(m,) 可选。
    四道闸：
      1) OOF AUC 下限（太弱）与上限（太高 = 泄漏/过拟合，反泄漏关键）；
      2) agreement gap：OOF 与 test 对基线（base_idx 中位数）的相关性差过大；
      3) mean drift：test 与 OOF 的概率均值漂移过大；
      4) 近重复：与已留成员的 rank 相关 > dup_rho 去重（按 OOF AUC 降序贪心）。
    """
    OOF = np.clip(np.asarray(OOF, float), -CLIP, CLIP)
    TEST = np.clip(np.asarray(TEST, float), -CLIP, CLIP)
    n, m = OOF.shape
    y = np.asarray(y)
    if names is None:
        names = [f"m{j}" for j in range(m)]
    oof_auc = np.array([roc_auc_score(y, OOF[:, j]) for j in range(m)])
    if base_idx is None:
        base_idx = np.arange(m)
    ref_o = np.median(OOF[:, base_idx], axis=1)
    ref_t = np.median(TEST[:, base_idx], axis=1)
    agree_oof = _corr_cols(OOF, ref_o)
    agree_test = _corr_cols(TEST, ref_t)
    p_o = 1 / (1 + np.exp(-OOF))
    p_t = 1 / (1 + np.exp(-TEST))
    drift_vals = np.abs(p_t.mean(0) - p_o.mean(0))

    status = [""] * m
    keep = np.ones(m, bool)
    for j in range(m):
        if oof_auc[j] < min_auc:
            status[j] = f"OOF AUC {oof_auc[j]:.4f} too low"; keep[j] = False
        elif oof_auc[j] > max_auc:
            status[j] = f"OOF AUC {oof_auc[j]:.4f} too high (leak/overfit)"; keep[j] = False
        elif agree_oof[j] - agree_test[j] > agree_gap:
            status[j] = "agreement gap"; keep[j] = False
        elif drift_vals[j] > drift:
            status[j] = "mean drift"; keep[j] = False

    # 近重复去重：按 OOF AUC 降序贪心保留
    R = np.column_stack([rankdata(OOF[:, j]) for j in range(m)]).astype(float)
    R -= R.mean(0)                        # 先中心化
    R /= np.sqrt((R ** 2).mean(0))        # 再除 std（此时 R**2 已是中心化后的）
    kept_cols = []
    for j in sorted(range(m), key=lambda i: -oof_auc[i]):
        if not keep[j]:
            continue
        dup = False
        for k in kept_cols:
            rho = float((R[:, j] @ R[:, k]) / n)
            if rho > dup_rho:
                status[j] = f"near-dup rho {rho:.6f} vs {names[k]}"
                keep[j] = False
                dup = True
                break
        if not dup:
            kept_cols.append(j)

    report = pd.DataFrame(dict(name=names, oof_auc=oof_auc.round(6),
                               agree_oof=agree_oof.round(6), agree_test=agree_test.round(6),
                               drift=drift_vals.round(6), status=status))
    return keep, report


# ---------- 5. 诚实选型（嵌套 CV + 跨种子） ----------

def honest_select(schemes, X, y, n_splits=5, seeds=(42, 7, 101)):
    """诚实选 stacking 方案：嵌套 CV + 跨种子复验，不碰公榜/OOF 搜权。

    schemes：{name: fn}，fn(X, y, folds) -> oof 向量（内部须做嵌套 CV，见 nested_lr_cv）。
    对每个 seed 重建 folds、跑每个 scheme，返回 (best_name, summary, report)：
      summary：每 scheme 的 fold-mean 均值/std/min/max（跨种子）；
      report ：逐 (scheme, seed) 的 fold-mean。
    """
    rows = []
    for seed in seeds:
        folds = make_folds(y, n_splits, seed)
        for name, fn in schemes.items():
            oof = fn(X, y, folds)
            rows.append(dict(seed=seed, scheme=name, fold_mean=fold_mean_auc(y, oof, folds)))
    report = pd.DataFrame(rows)
    summary = report.groupby("scheme")["fold_mean"].agg(mean="mean", std="std", min="min", max="max")
    best = summary["mean"].idxmax()
    return best, summary, report


# ---------- 可选：LGB stacker（懒加载 lightgbm） ----------

def lgb_stacker_cv(X, y, folds, params=None):
    """嵌套 LGB stacker（raw_score），返回 OOF。仅调用时才 import lightgbm。"""
    import lightgbm as lgb
    p = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=2000,
             feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
             verbose=-1, num_threads=4, seed=0)
    if params:
        p.update(params)
    o = np.zeros(len(y), float)
    for tr, va in folds:
        m = lgb.train(p, lgb.Dataset(X[tr], y[tr]), 3000,
                      valid_sets=[lgb.Dataset(X[va], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        o[va] = m.predict(X[va], raw_score=True, num_iteration=m.best_iteration)
    return o


# ---------- 自检 demo（无真实数据，跑通 API 用） ----------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, n_test, m = 4000, 1000, 6
    x = rng.normal(size=(n, 5))
    xt = rng.normal(size=(n_test, 5))                  # test 侧特征
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=1.5, size=n) > 0).astype(int)

    sig = 0.6 * x[:, 0] + 0.3 * x[:, 1]
    sigt = 0.6 * xt[:, 0] + 0.3 * xt[:, 1]
    OOF = np.column_stack([
        sig + rng.normal(scale=1.0, size=n),            # 0 好
        0.4 * x[:, 0] + rng.normal(scale=1.2, size=n),  # 1 一般
        y.astype(float) * 8 - 4,                        # 2 泄漏：AUC 过高
        sig + rng.normal(scale=0.001, size=n),          # 3 近重复 0
        0.5 * x[:, 1] + rng.normal(scale=1.0, size=n),  # 4 正常但 test 侧漂移
        rng.normal(size=n),                             # 5 垃圾
    ])
    TEST = np.column_stack([
        sigt + rng.normal(scale=1.0, size=n_test),      # 0
        0.4 * xt[:, 0] + rng.normal(scale=1.2, size=n_test),  # 1
        rng.normal(size=n_test),                        # 2 泄漏：test 无信号
        sigt + rng.normal(scale=0.001, size=n_test),    # 3
        0.5 * xt[:, 1] + rng.normal(scale=1.0, size=n_test) + 2.0,  # 4 漂移
        rng.normal(size=n_test),                        # 5
    ])
    names = ["lgbm_good", "xgb_ok", "leaky", "lgbm_dup", "cat_drift", "junk"]

    print("=== QC 四件套（demo 阈值 0.55/0.85）===")
    keep, rep = qc_members(OOF, TEST, y, names, min_auc=0.55, max_auc=0.85)
    print(rep.to_string(index=False))
    print("keep:", [names[i] for i in range(m) if keep[i]])

    print("\n=== 分族缩放 ===")
    fam, sc = family_scale(["tabpfn35", "realmlp_v4", "xgb_te", "tfm_cat"])
    for f, s in zip(fam, sc):
        print(f"{f:10s} -> x{s}")

    print("\n=== 诚实选型 flat vs family ===")
    kept_names = [names[i] for i in range(m) if keep[i]]
    Xo = OOF[:, keep]
    _, scales = family_scale(kept_names)
    X_fam = Xo * scales
    schemes = {
        "flat_C003":   lambda X, y, f: nested_lr_cv(X, y, f, C=0.003),
        "family_C025": lambda X, y, f: nested_lr_cv(X_fam, y, f, C=0.025),
    }
    best, summary, report = honest_select(schemes, Xo, y, seeds=(42, 7))
    print(summary.to_string())
    print("best:", best)
