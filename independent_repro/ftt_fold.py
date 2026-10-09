"""FT-Transformer 一折：内层选轮数 → 外层只评分 → 10% logit 混合 Busy53。

精简输入 = 原始 21 特征 + 航程类别表示（Flight Distance 类别副本）。
fold 0（5 折 seed 33，对齐 Busy53 的 oof53）。内层 80/20 早停选轮，外层 val 只评分。
"""
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pytabkit import FTT_D_Classifier

train = pd.read_csv("data/playground-series-s6e10/train.csv")
y = train["satisfaction"].to_numpy(np.int64)
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]


def prep(df):
    X = df.drop(columns=["id", "satisfaction"], errors="ignore").copy()
    X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(0.0)
    for c in CATS:
        X[c] = X[c].astype("category")
    X["Flight Distance_cat"] = X["Flight Distance"].astype(str).astype("category")
    return X


X = prep(train)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

folds = list(StratifiedKFold(5, shuffle=True, random_state=33).split(X, y))
ia, iv = folds[0]
print(f"fold 0: train {len(ia)} / val {len(iv)}", flush=True)

# 内层 80/20 选轮数
inner = StratifiedKFold(5, shuffle=True, random_state=0)
i_tr, i_va = next(iter(inner.split(X.iloc[ia], y[ia])))

t0 = time.time()
m = FTT_D_Classifier(max_epochs=30, es_patience=5, batch_size=256, device="cuda", verbosity=0)
m.fit(X.iloc[ia].iloc[i_tr], y[ia][i_tr], X.iloc[ia].iloc[i_va], y[ia][i_va])
pred = m.predict_proba(X.iloc[iv])[:, 1]
print(f"fold 0 单模型 AUC = {roc_auc_score(y[iv], pred):.6f}  ({time.time()-t0:.0f}s)", flush=True)

np.savez("cache/ftt_fold0.npz", val_id=train["id"].iloc[iv].to_numpy(), pred=pred, y_val=y[iv])

# 10% logit 混合 Busy53
oof53 = np.load("pulled_busy53/oof_stacks.npz")["oof53"]  # 已是 logit（decision_function）
base = oof53[iv]
blend = 0.9 * base + 0.1 * logit(pred)
print(f"Busy53(fold0) = {roc_auc_score(y[iv], base):.6f}", flush=True)
print(f"+10% FTT      = {roc_auc_score(y[iv], blend):.6f}  Δ {roc_auc_score(y[iv], blend) - roc_auc_score(y[iv], base):+.6f}", flush=True)
