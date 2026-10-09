"""TabR（适配版）单折互补性实验：fold 0，内层选轮数，10% logit 混合 Busy53。"""
import sys
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, r"C:/Users/GEM07/Documents/Codex/2026-10-02/https-www-kaggle-com-competitions-playground/outputs/tabr_compat_20261009")
from tabr_compat import CompatibleTabRClassifier

torch.set_num_threads(4)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False

train = pd.read_csv("data/playground-series-s6e10/train.csv")
y = train["satisfaction"].to_numpy(np.int64)


def input_frame(df):
    x = df.drop(columns=["id", "satisfaction"], errors="ignore").copy()
    x["Arrival Delay in Minutes"] = x["Arrival Delay in Minutes"].fillna(0.0)
    for c in ["Gender", "Customer Type", "Type of Travel", "Class"]:
        x[c] = x[c].astype("category")
    x["Flight Distance_cat"] = (x["Flight Distance"] // 250).astype(str).astype("category")
    return x


X = input_frame(train)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

folds = list(StratifiedKFold(5, shuffle=True, random_state=33).split(X, y))
ia, iv = folds[0]
print(f"fold 0: train {len(ia)} / val {len(iv)}", flush=True)

# 内层 80/20 选轮数
inner = StratifiedKFold(5, shuffle=True, random_state=0)
i_tr, i_va = next(iter(inner.split(X.iloc[ia], y[ia])))

model = CompatibleTabRClassifier(
    n_epochs=30, patience=10, batch_size=128, eval_batch_size=128,
    context_size=96, memory_efficient=True, candidate_encoding_batch_size=2048,
    device="cuda", n_threads=4, random_state=42, verbosity=0,
    val_metric_name="cross_entropy", tmp_folder="cache/tabr_tmp")

t0 = time.time()
model.fit(X.iloc[ia].iloc[i_tr], y[ia][i_tr], X.iloc[ia].iloc[i_va], y[ia][i_va])
pred = model.predict_proba(X.iloc[iv])[:, 1]
print(f"TabR fold0 单模型 AUC = {roc_auc_score(y[iv], pred):.6f}  ({time.time()-t0:.0f}s)", flush=True)

np.savez("cache/tabr_fold0.npz", val_id=train["id"].iloc[iv].to_numpy(), pred=pred, y_val=y[iv])

oof53 = np.load("pulled_busy53/oof_stacks.npz")["oof53"]
base = oof53[iv]
blend = 0.9 * base + 0.1 * logit(pred)
print(f"Busy53(fold0) = {roc_auc_score(y[iv], base):.6f}", flush=True)
print(f"+10% TabR      = {roc_auc_score(y[iv], blend):.6f}  Δ {roc_auc_score(y[iv], blend) - roc_auc_score(y[iv], base):+.6f}", flush=True)
