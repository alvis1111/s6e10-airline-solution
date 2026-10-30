"""S6E10 — 计算 aux-task 期望值特征并存入缓存（供 route_profile_aux.py 用）。"""
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
ORIGINAL_FEATURES = NUM_COLS + RATING_COLS + CAT_COLS_RAW
AUX_COLS = [f"aux_ev_{c}" for c in RATING_COLS]

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
NTR = len(train)

full = pd.concat([train.drop(columns=["satisfaction"]), test], ignore_index=True)
coded = full.copy()
for c in CAT_COLS_RAW:
    coded[c] = coded[c].astype("category").cat.codes
coded["Arrival Delay in Minutes"] = coded["Arrival Delay in Minutes"].fillna(-1)

aux_ev = np.zeros((len(full), len(RATING_COLS)), dtype=np.float32)
t0 = time.time()
for j, col in enumerate(RATING_COLS):
    yc, vals = pd.factorize(coded[col], sort=True)
    vals = np.asarray(vals, dtype=np.float32)
    k = int(yc.max()) + 1
    others = [f for f in ORIGINAL_FEATURES if f != col]
    P = np.zeros((len(full), k), dtype=np.float32)
    for a, b in KFold(5, shuffle=True, random_state=0).split(coded):
        m = xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.7, tree_method="hist",
            objective="multi:softprob", num_class=k, n_jobs=-1, verbosity=0, random_state=0)
        m.fit(coded.iloc[a][others], yc[a])
        P[b] = m.predict_proba(coded.iloc[b][others])
    aux_ev[:, j] = (P * vals).sum(axis=1)
    print(f"  {col:38s} {time.time()-t0:5.0f}s", flush=True)

np.save("cache/aux_ev_train.npy", aux_ev[:NTR])
np.save("cache/aux_ev_test.npy", aux_ev[NTR:])
print(f"已保存 cache/aux_ev_train.npy ({aux_ev[:NTR].shape}) / cache/aux_ev_test.npy ({aux_ev[NTR:].shape})，总用时 {time.time()-t0:.0f}s", flush=True)
