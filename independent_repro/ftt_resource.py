"""FT-Transformer 资源短跑：小上下文 + 2 轮，测 8GB 显存与速度。"""
import time
import numpy as np
import pandas as pd
import torch
from pytabkit import FTT_D_Classifier

train = pd.read_csv("data/playground-series-s6e10/train.csv")
test = pd.read_csv("data/playground-series-s6e10/test.csv")

CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]

# 精简输入：原始 21 特征 + 航程类别表示（Flight Distance 的类别副本）
def prep(df):
    X = df.drop(columns=["id", "satisfaction"], errors="ignore").copy()
    X["Arrival Delay in Minutes"] = X["Arrival Delay in Minutes"].fillna(0.0)
    for c in CATS:
        X[c] = X[c].astype("category")
    # 航程类别表示（类别 dtype，与数值 Flight Distance 并存）
    X["Flight Distance_cat"] = X["Flight Distance"].astype(str).astype("category")
    return X

Xtr = prep(train)
Xte = prep(test)
y = train["satisfaction"].to_numpy(np.int64)
print(f"特征 {Xtr.shape[1]} 列", flush=True)

# 小上下文测资源
N = 50000
Xa = Xtr.iloc[:N]; ya = y[:N]
m = FTT_D_Classifier(max_epochs=2, batch_size=256, device="cuda", verbosity=1)

t0 = time.time()
m.fit(Xa, ya)
dt = time.time() - t0
print(f"fit {N} 行 x 2 轮 = {dt:.0f}s", flush=True)
if torch.cuda.is_available():
    print(f"显存峰值 {torch.cuda.max_memory_allocated()/2**30:.2f} GB", flush=True)

p = m.predict_proba(Xtr.iloc[:5000])[:, 1]
from sklearn.metrics import roc_auc_score
print(f"AUC(前5000行, 训练集) = {roc_auc_score(y[:5000], p):.5f}", flush=True)
