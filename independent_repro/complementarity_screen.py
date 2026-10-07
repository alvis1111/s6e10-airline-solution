"""Step 3 互补性初筛（fold 0 版）：独立复现的 TabPFN 加进自有融合，是否有稳定增益。

协议（呼应「先验证互补性，再扩折」）：
  1. 基线 = 自有融合（route_aux + context + tabm 坐标下降权重，rank-blend）。
  2. 只取 fold 0 的验证行（TabPFN 在这些行上是诚实的 OOF），split seed=20261006 分成
     探索半集（选 ε）与确认半集（评一次）。
  3. 探索半集上从固定小权重集 {0.005,0.01,0.02,0.03,0.05} 选最佳 ε；
     确认半集上用配对 DeLong 评「baseline + ε·tabpfn」vs baseline 的 AUC 差区间。
  4. 通过条件：确认半集 delta ≥ +0.00002 且 95% 区间下界 > 0（记忆里的预设筛选）。
     两个变体（raw / catfd）都满足才投入完整 5 折 + 提交。

注意：full-OOF 坐标下降的基线权重略有高估（见记忆「OOF 搜权高估」），
这里基线是固定函数、ε 在探索半集选，所以 TabPFN 的增益测量本身是诚实的。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata

from delong import delong_ci

EPS = [0.005, 0.01, 0.02, 0.03, 0.05]
SPLIT_SEED = 20261006
THRESH = 0.00002


def main():
    train = pd.read_csv("data/playground-series-s6e10/train.csv")
    y = train["satisfaction"].to_numpy(np.int64)
    n = len(train)

    # --- 自有融合基线（rank-blend，坐标下降权重） ---
    ra = 0.5 * np.load("cache/oof_route_aux_l.npy") + 0.5 * np.load("cache/oof_route_aux_x.npy")
    tb = np.load("cache/oof_route_tabm.npy")
    z = np.load(r"C:/Users/GEM07/OneDrive/Desktop/s6e10_outer10_blend/previous_context_ensemble.npz")
    order = {tid: i for i, tid in enumerate(train["id"].to_numpy())}
    ctx = np.full(n, np.nan)
    ctx[[order[t] for t in z["train_id"]]] = z["oof"]
    assert not np.isnan(ctx).any(), "context alignment failed"

    r = lambda p: rankdata(p) / len(p)
    R = np.column_stack([r(ra), r(ctx), r(tb)])
    W = np.array([0.516, 0.387, 0.097])
    base = R @ W
    print(f"自有融合基线 OOF = {roc_auc_score(y, base):.6f}", flush=True)

    # --- fold 0 验证行 ---
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(n), y))
    b = folds[0][1]
    print(f"fold 0 验证行数 = {len(b)}", flush=True)

    # --- 探索 / 确认 半集（在 fold 0 验证行内再划） ---
    inner = list(StratifiedKFold(2, shuffle=True, random_state=SPLIT_SEED).split(np.zeros(len(b)), y[b]))
    exp_idx, conf_idx = inner[0][0], inner[0][1]  # 探索 = 折0训练，确认 = 折0验证
    print(f"探索 {len(exp_idx)} / 确认 {len(conf_idx)} 行", flush=True)

    for variant in ["raw", "catfd"]:
        d = np.load(f"cache/partial/tabpfn_repro_{variant}_f0.npz")
        oof_v = d["oof"]                      # 长度 = len(b)，与 b 顺序一致
        auc_v = roc_auc_score(y[b], oof_v)
        print(f"\n=== {variant}  单腿 fold0 AUC = {auc_v:.6f} ===", flush=True)

        # 探索半集选 ε
        best = (None, -1.0)
        for eps in EPS:
            new = (1 - eps) * base[b] + eps * r(oof_v)
            a = roc_auc_score(y[b][exp_idx], new[exp_idx])
            if a > best[1]:
                best = (eps, a)
        eps, exp_auc = best
        print(f"探索半集最优 ε = {eps}  AUC = {exp_auc:.6f}", flush=True)

        # 确认半集评一次（DeLong）
        new_conf = (1 - eps) * base[b] + eps * r(oof_v)
        _, _, delta, lo, hi, p = delong_ci(y[b][conf_idx], base[b][conf_idx], new_conf[conf_idx])
        passed = delta >= THRESH and lo > 0
        print(f"确认半集: 基线 {roc_auc_score(y[b][conf_idx], base[b][conf_idx]):.6f} -> "
              f"+tabpfn {roc_auc_score(y[b][conf_idx], new_conf[conf_idx]):.6f}  "
              f"delta={delta:+.7f}  95%CI=[{lo:+.7f}, {hi:+.7f}]  p={p:.4f}  "
              f"{'通过' if passed else '未通过'}", flush=True)


if __name__ == "__main__":
    main()
