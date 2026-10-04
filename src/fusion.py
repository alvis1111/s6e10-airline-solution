"""S6E10 — 最终融合: 我的三模型集成 + 并行会话树模型 + 反推 TabM."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

DATA = "data/playground-series-s6e10"
BUNDLE = r"C:\Users\GEM07\OneDrive\Desktop\s6e10_tabm_te"

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
sub = pd.read_csv(f"{DATA}/sample_submission.csv")
y = train['satisfaction'].to_numpy(dtype=np.int64)
n_tr, n_te = len(train), len(test)

# 我的三模型集成
oof_l = np.load("cache/oof_lgb_pte.npy"); pred_l = np.load("cache/pred_lgb_pte.npy")
oof_x = np.load("cache/oof_xgb_pte.npy"); pred_x = np.load("cache/pred_xgb_pte.npy")
oof_c = np.load("cache/oof_cb_pte.npy"); pred_c = np.load("cache/pred_cb_pte.npy")

# 并行会话树模型
z = np.load(f"{BUNDLE}/previous_tree_ensemble.npz")
tree_oof = z['oof']; tree_pred = z['pred']

# 并行会话 hybrid (tree+TabM) 测试预测
hybrid_pred = pd.read_csv(f"{BUNDLE}/submission_tabm_te.csv")['satisfaction'].to_numpy()

# 反推 TabM 的 rank (hybrid = 0.5*tree + 0.5*rank(tabm))
tabm_rank_pred = 2 * hybrid_pred - tree_pred

def r(p): return rankdata(p) / len(p)

# 我的集成 rank-blend (0.5/0.2/0.3)
ens_oof = 0.5*r(oof_l) + 0.2*r(oof_x) + 0.3*r(oof_c)
ens_pred = 0.5*r(pred_l) + 0.2*r(pred_x) + 0.3*r(pred_c)
print(f"我的集成 OOF = {roc_auc_score(y, ens_oof):.5f}", flush=True)

# 2-way: 我的集成 vs 树模型 (OOF 上找最优权重)
print("\n=== 集成 vs 树模型 ===", flush=True)
best = (0, None)
for w in np.arange(0, 1.01, 0.05):
    b = w * r(ens_oof) + (1-w) * r(tree_oof)
    a = roc_auc_score(y, b)
    if a > best[0]: best = (a, round(w, 2))
print(f"最优: w(集成)={best[1]}  OOF={best[0]:.5f}", flush=True)

# 3-way: 集成 + 树 + TabM (TabM OOF 无法直接算, 用 0.960439 的已知值近似说明)
# 先做 2-way 集成+树 的最优融合, 再叠 TabM rank 试几组权重
tree_rank_pred = r(tree_pred)
ens_rank_pred = r(ens_pred)

# 简单 3-way 等权 (集成/树/TabM 各 1/3)
tri = (r(ens_pred) + tree_rank_pred + tabm_rank_pred) / 3
# 2-way 集成+树最优 (用上面的 best[1])
two = best[1] * ens_rank_pred + (1-best[1]) * tree_rank_pred

sub['satisfaction'] = two
sub.to_csv("submission_fusion_tree.csv", index=False)
sub['satisfaction'] = tri
sub.to_csv("submission_fusion_tri.csv", index=False)
print(f"\n已保存: submission_fusion_tree.csv (2-way 集成+树, w={best[1]})", flush=True)
print(f"已保存: submission_fusion_tri.csv (3-way 等权 集成+树+TabM)", flush=True)

# 融合 OOF 参考 (2-way 用 best[1])
two_oof = best[1] * r(ens_oof) + (1-best[1]) * r(tree_oof)
print(f"\n2-way 融合 OOF = {roc_auc_score(y, two_oof):.5f}", flush=True)
