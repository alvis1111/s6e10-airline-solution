"""复现 Golem 库 README 的三个基线数字，验证成员对齐、行序、预测尺度与融合配方。

- 21 单成员 plain logistic 堆叠 nested AUC 应为 0.961860
- stack247_oof.npy 单独 AUC 应为 0.962205
- stack247 + 4 个 v6 新成员 AUC 应为 0.962198
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA = "data/playground-series-s6e10"
G = "pulled_golem"


def _to_label(v) -> int:
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    s = str(v).strip().lower()
    if s in ("true", "1", "1.0", "satisfied"):
        return 1
    if s in ("false", "0", "0.0", "neutral or dissatisfied", "dissatisfied"):
        return 0
    return int(v)


def main() -> None:
    train = pd.read_csv(f"{DATA}/train.csv")
    print(f"train shape={train.shape}  cols={list(train.columns)[:8]}")
    print(f"satisfaction dtype={train['satisfaction'].dtype}  unique={train['satisfaction'].unique()[:6]}")

    oof = pd.read_parquet(f"{G}/oof_wide.parquet")
    test = pd.read_parquet(f"{G}/test_wide.parquet")
    print(f"oof_wide={oof.shape}  test_wide={test.shape}")
    print(f"oof_wide cols={list(oof.columns)}")

    y = train.set_index("id").loc[oof["id"], "satisfaction"].map(_to_label).to_numpy()
    print(f"positive rate={y.mean():.6f}  (expect 0.443573)")

    cols = [c for c in oof.columns if c not in ("id", "fold")]
    print(f"member count={len(cols)}")

    lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
    X = lg(oof[cols].to_numpy())
    XT = lg(test[cols].to_numpy())
    fold = oof["fold"].to_numpy()

    # 1) 21-member plain logistic stack
    stack = np.zeros(len(oof))
    sub = np.zeros(len(test))
    for k in range(5):
        m = LogisticRegression(C=1.0, max_iter=2000).fit(X[fold != k], y[fold != k])
        stack[fold == k] = m.decision_function(X[fold == k])
        sub = sub + m.predict_proba(XT)[:, 1] / 5
    print(f"\n21-stack nested AUC = {roc_auc_score(y, stack):.6f}   (expect 0.961860)")

    # 2) stack247 alone
    s247 = np.load(f"{G}/stack247_oof.npy")
    s247t = np.load(f"{G}/stack247_test.npy")
    print(f"stack247_oof AUC   = {roc_auc_score(y, s247):.6f}   (expect 0.962205)")
    print(f"stack247 shapes oof={s247.shape} test={s247t.shape}")

    # 3) stack247 + version-6 members (they were never among the 247 inputs)
    new = ["r_realmlp_auxev", "s_realmlp_raw_auxev_10fold", "t_xgb_d5_auxev", "u_catboost_ctr_auxev"]
    Z = np.column_stack([lg(s247), lg(oof[new].to_numpy())])
    ZT = np.column_stack([lg(s247t), lg(test[new].to_numpy())])
    stack2 = np.zeros(len(oof))
    for k in range(5):
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Z[fold != k], y[fold != k])
        stack2[fold == k] = m.decision_function(Z[fold == k])
    print(f"stack247 + new AUC = {roc_auc_score(y, stack2):.6f}   (expect 0.962198)")

    # 4) spot-check single-member AUCs against manifest
    print("\nspot-check single members (vs manifest oof_auc):")
    for c, expected in [("a_lgbm_digits", 0.959568), ("n_realmlp", 0.961233),
                        ("o_tabpfn35", 0.961082), ("q_lgbm_auxev_10fold", 0.961388)]:
        print(f"  {c:22s} = {roc_auc_score(y, oof[c].to_numpy()):.6f}  (expect {expected})")


if __name__ == "__main__":
    main()
