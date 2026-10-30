"""步骤 2：把自有 lean RealMLP 叠到 stack247 / 21 成员上，配对 DeLong 测增量。

冻结 5 折 (seed 42) 嵌套 logistic 堆叠，与 README 一致：logits、C=1.0、max_iter 2000。
增量口径：5 折配对 fold 差（均值/SE/正折数）+ 整体配对 DeLong。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from delong import delong_ci

DATA = "data/playground-series-s6e10"
G = "pulled_golem"

train = pd.read_csv(f"{DATA}/train.csv")
y = train["satisfaction"].astype(int).to_numpy()

oof = pd.read_parquet(f"{G}/oof_wide.parquet")
fold = oof["fold"].to_numpy()
cols = [c for c in oof.columns if c not in ("id", "fold")]

lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))

s247 = lg(np.load(f"{G}/stack247_oof.npy"))
X21 = lg(oof[cols].to_numpy())
own_prob = np.load("cache/oof_realmlp_lean.npy")
own = lg(own_prob)

print(f"lean RealMLP own AUC = {roc_auc_score(y, own_prob):.6f}")


def nested_stack(parts: list[np.ndarray]) -> tuple[np.ndarray, list[float]]:
    Z = np.column_stack(parts)
    out = np.zeros(len(y))
    fold_aucs = []
    for k in range(5):
        m = LogisticRegression(C=1.0, max_iter=2000).fit(Z[fold != k], y[fold != k])
        out[fold == k] = m.decision_function(Z[fold == k])
        fold_aucs.append(roc_auc_score(y[fold == k], out[fold == k]))
    return out, fold_aucs


def compare(name: str, base_parts: list[np.ndarray], cand_parts: list[np.ndarray]) -> None:
    base, fa = nested_stack(base_parts)
    cand, fb = nested_stack(cand_parts)
    diffs = np.array(fb) - np.array(fa)
    se = diffs.std(ddof=1) / np.sqrt(5)
    _, _, delta, lo, hi, p = delong_ci(y, base, cand)
    print(f"\n[{name}]")
    print(f"  baseline  OOF = {roc_auc_score(y, base):.6f}")
    print(f"  + lean    OOF = {roc_auc_score(y, cand):.6f}")
    print(f"  per-fold Δ = {[f'{d:+.6f}' for d in diffs]}")
    print(f"  mean Δ = {diffs.mean():+.6f}  SE = {se:.6f}  folds+ = {(diffs > 0).sum()}/5")
    print(f"  DeLong Δ = {delta:+.7f}  CI=[{lo:+.7f}, {hi:+.7f}]  p={p:.4f}")


compare("stack247 vs stack247+lean", [s247], [s247, own])
compare("21-stack vs 21-stack+lean", [X21], [X21, own])

# 全量：stack247 + 21 成员 + lean
full, _ = nested_stack([s247, X21, own])
print(f"\n[full] stack247 + 21 + lean OOF = {roc_auc_score(y, full):.6f}")
