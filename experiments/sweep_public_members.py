"""sweep_public_members — 查漏公开基础模型预测(30分钟上限)。

对照 F26 fold-0 基线, 遍历已下载的公开 OOF(Naji/megayak/cdeotte其他TFM/goodpjw2008),
算每个候选的融合增量 0.9*r(F26)+0.1*r(候选), 找真正没用过的、有增益的新成员。
"""
import numpy as np, pandas as pd, os
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HANDOFF = r"C:\Users\GEM07\Documents\Codex\2026-10-02\https-www-kaggle-com-competitions-playground\outputs\transfer_ensemble_handoff"
base = np.load(os.path.join(HANDOFF, "fold0_baseline.npz"), allow_pickle=False)
val_id = base["val_id"]; y = base["y_val"]; f26 = base["base_096126"]; row_idx = base["row_index"]
auc_base = roc_auc_score(y, f26)
rk = lambda x: rankdata(x, "average") / len(x)

cands = {}

# Naji (id 对齐)
naji = r"C:\Users\GEM07\.cache\kagglehub\datasets\najiama\s6e10-oof\versions\9"
for fn in os.listdir(naji):
    if fn.endswith("_oof.csv"):
        df = pd.read_csv(os.path.join(naji, fn))
        col = "satisfaction" if "satisfaction" in df.columns else df.columns[-1]
        cands[f"naji_{fn.replace('_oof.csv','')}"] = df.set_index("id").reindex(val_id)[col].to_numpy()

# megayak (id 对齐)
mg = r"C:\Users\GEM07\.cache\kagglehub\datasets\megayak\s6e10-oof-library\versions\3"
oof_mg = pd.read_parquet(os.path.join(mg, "oof_predictions.parquet")).set_index("id").reindex(val_id)
for c in oof_mg.columns:
    if c not in ("fold", "satisfaction"):
        cands[f"megayak_{c}"] = oof_mg[c].to_numpy()

# cdeotte 其他 TFM (行序对齐)
ct = r"C:\Users\GEM07\.cache\kagglehub\datasets\cdeotte\s6e10-tfm-oof-and-test-predictions\versions\1"
for fn in os.listdir(ct):
    if fn.startswith("oof_") and fn.endswith(".npy"):
        name = fn.replace("oof_","").replace(".npy","")
        if name == "tabpfn35":
            continue  # 已在 segment_fusion 用过
        cands[f"cdeotte_{name}"] = np.load(os.path.join(ct, fn))[row_idx]

# goodpjw2008 (id 对齐; 排除已在 segment_fusion 用过的 tabpfn 变体)
gp = r"C:\Users\GEM07\.cache\kagglehub\datasets\goodpjw2008\s6e10-tabpfn-member\versions\2"
used_gp = {"tabpfn_te4","tabpfn_te","tabpfn_catfd10","tabpfn_catfd","tabpfn_full_raw","tabpfn_full_raw_s1","tabpfn_catfd_s1"}
for fn in os.listdir(gp):
    if fn.startswith("oof_") and fn.endswith(".csv"):
        name = fn.replace("oof_","").replace(".csv","")
        if name in used_gp:
            continue
        df = pd.read_csv(os.path.join(gp, fn))
        cands[f"goodpjw2008_{name}"] = df.set_index("id").reindex(val_id)["pred"].to_numpy()

print(f"基线 F26 AUC = {auc_base:.6f}, 候选成员数 = {len(cands)}", flush=True)
rows = []
for name, pred in cands.items():
    if len(pred) != len(y) or not np.isfinite(pred).all():
        continue
    pred = np.clip(pred, 1e-6, 1-1e-6)
    auc_single = roc_auc_score(y, pred)
    blend = 0.9*rk(f26) + 0.1*rk(pred)
    auc_blend = roc_auc_score(y, blend)
    corr = np.corrcoef(pred, f26)[0,1]
    rows.append((name, auc_single, auc_blend, auc_blend-auc_base, corr))

rows.sort(key=lambda r: -r[3])
print(f"\n{'成员':<28} {'单模型AUC':>10} {'融合AUC':>10} {'Δ融合':>9} {'corr':>8}")
for name, s, b, d, c in rows[:20]:
    print(f"{name:<28} {s:>10.6f} {b:>10.6f} {d:>+9.6f} {c:>8.5f}")
top = rows[0] if rows else None
print(f"\n最强候选: {top[0] if top else '无'}, Δ融合={top[3]:+.6f} (门槛 ~+0.0001)" if top else "")
