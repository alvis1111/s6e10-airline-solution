"""S6E10 — 补全 aux 特征到 30 列：13 ev + 16 观测值概率(aux_p) + 1 行 log-likelihood(aux_logp)。

goodpjw2008 Part4 完整定义：对 13 评分 + Class/Type of Travel/Customer Type 共 16 列，
各训「用其余 20 列预测该列」的多分类 XGB（无标签、5 折），取 aux_p = P(实际值)；
aux_logp = Σ log(aux_p) 衡量整行一致性。aux_ev(13) 已在 cache/aux_ev_*.npy。
本脚本补 aux_p(16) + aux_logp(1)。
"""
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold

DATA = "data/playground-series-s6e10"
RATING_COLS = ["Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
    "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness"]
CAT_COLS_RAW = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
FEATS = NUM_COLS + RATING_COLS + CAT_COLS_RAW
AUX_TARGETS = RATING_COLS + ["Class", "Type of Travel", "Customer Type"]   # 16 列

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
NTR = len(train)
full = pd.concat([train.drop(columns=["satisfaction"]), test], ignore_index=True)
coded = full.copy()
for c in CAT_COLS_RAW:
    coded[c] = coded[c].astype("category").cat.codes
coded["Arrival Delay in Minutes"] = coded["Arrival Delay in Minutes"].fillna(-1)

aux_p = np.zeros((len(full), len(AUX_TARGETS)), dtype=np.float32)
t0 = time.time()
for j, col in enumerate(AUX_TARGETS):
    yc, vals = pd.factorize(coded[col], sort=True)
    k = int(yc.max()) + 1
    others = [f for f in FEATS if f != col]
    P = np.zeros((len(full), k), dtype=np.float32)
    for a, b in KFold(5, shuffle=True, random_state=0).split(coded):
        m = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.7, tree_method="hist",
            objective="multi:softprob", num_class=k, n_jobs=-1, verbosity=0, random_state=0)
        m.fit(coded.iloc[a][others], yc[a])
        P[b] = m.predict_proba(coded.iloc[b][others])
    aux_p[:, j] = P[np.arange(len(full)), yc]          # P(实际值)
    print(f"  {col:38s} {time.time()-t0:5.0f}s  aux_p 中位={np.median(aux_p[:,j]):.3f}", flush=True)

aux_logp = np.log(np.clip(aux_p, 1e-9, None)).sum(axis=1).astype(np.float32)

np.save("cache/aux_p_train.npy", aux_p[:NTR]); np.save("cache/aux_p_test.npy", aux_p[NTR:])
np.save("cache/aux_logp_train.npy", aux_logp[:NTR]); np.save("cache/aux_logp_test.npy", aux_logp[NTR:])
print(f"已保存 aux_p(16) + aux_logp(1)，总用时 {time.time()-t0:.0f}s", flush=True)
