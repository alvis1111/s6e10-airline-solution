"""S6E10 — proper TE (smooth auto + 100) + LGBM/XGB/CatBoost 集成."""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

DATA = "data/playground-series-s6e10"
CATS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
SERVICES = ['Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding',
    'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Cleanliness']

def build(df):
    X = df.drop(columns=['id', 'satisfaction'], errors='ignore').copy()
    X['traveler_segment'] = X['Type of Travel'] + '|' + X['Customer Type'] + '|' + X['Class']
    X['arrival_missing'] = X['Arrival Delay in Minutes'].isna().astype('float32')
    for c in CATS + SERVICES + ['traveler_segment']:
        X[c] = X[c].astype(str)
    return X

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train['satisfaction'].to_numpy(dtype=np.int64)
X, T = build(train), build(test)

NUM = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes', 'arrival_missing']
CAT_STR = CATS + SERVICES + ['traveler_segment']

# 类别编码器在全量 train+test 上预 fit (无目标信息, 避免罕见组合 unseen)
LES = {}
for c in CAT_STR:
    le = LabelEncoder().fit(pd.concat([X[c], T[c]]).astype(str))
    LES[c] = le

# 每折: TE (smooth auto + 100) + label-encode 类别
def fold_matrices(tr, va):
    Xt, Xv, Xte = X.iloc[tr].copy(), X.iloc[va].copy(), T.copy()
    # 数值
    for f in [Xt, Xv, Xte]:
        for c in NUM:
            f[c] = f[c].fillna(f[c].median() if f is Xt else Xt[c].median()).astype('float32')
    # label-encode 类别 (全量预 fit)
    for c in CAT_STR:
        Xt[c] = LES[c].transform(Xt[c]); Xv[c] = LES[c].transform(Xv[c]); Xte[c] = LES[c].transform(Xte[c])
    # TE (两种平滑)
    Z = train.drop(columns=['id', 'satisfaction']).astype(str)
    U = test.drop(columns=['id']).astype(str)
    for smooth in ['auto', 100]:
        enc = TargetEncoder(target_type='binary', smooth=smooth, cv=5, shuffle=True, random_state=20261002)
        et = enc.fit_transform(Z.iloc[tr], y[tr]).astype('float32')
        ev = enc.transform(Z.iloc[va]).astype('float32')
        ee = enc.transform(U).astype('float32')
        tag = 'auto' if smooth == 'auto' else 's100'
        for f, encoded in [(Xt, et), (Xv, ev), (Xte, ee)]:
            for i in range(encoded.shape[1]):
                f[f'te_{tag}_{i}'] = encoded[:, i]
    return Xt, Xv, Xte

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof_l, pred_l = np.zeros(len(train)), np.zeros(len(test))
oof_x, pred_x = np.zeros(len(train)), np.zeros(len(test))
oof_c, pred_c = np.zeros(len(train)), np.zeros(len(test))

for fold, (tr, va) in enumerate(skf.split(X, y)):
    Xt, Xv, Xte = fold_matrices(tr, va)
    # LGBM
    m1 = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
        min_child_samples=50, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
        reg_alpha=0.3, reg_lambda=1.0, n_estimators=8000, random_state=42, n_jobs=-1, verbosity=-1)
    m1.fit(Xt, y[tr], eval_X=Xv, eval_y=y[va], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof_l[va] = m1.predict_proba(Xv)[:, 1]; pred_l += m1.predict_proba(Xte)[:, 1]/5
    # XGBoost
    m2 = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", learning_rate=0.03,
        max_depth=8, min_child_weight=20, colsample_bytree=0.8, subsample=0.9,
        reg_alpha=0.1, reg_lambda=1.0, n_estimators=8000, tree_method="hist",
        random_state=42, n_jobs=-1, early_stopping_rounds=150)
    m2.fit(Xt, y[tr], eval_set=[(Xv, y[va])], verbose=False)
    oof_x[va] = m2.predict_proba(Xv)[:, 1]; pred_x += m2.predict_proba(Xte)[:, 1]/5
    # CatBoost
    m3 = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", learning_rate=0.1,
        depth=6, l2_leaf_reg=3, iterations=3000, od_type="Iter", od_wait=100,
        verbose=False, thread_count=-1, random_seed=42)
    m3.fit(Xt, y[tr], eval_set=(Xv, y[va]), use_best_model=True)
    oof_c[va] = m3.predict_proba(Xv)[:, 1]; pred_c += m3.predict_proba(Xte)[:, 1]/5
    print(f"fold {fold}: LGB={roc_auc_score(y[va],oof_l[va]):.5f} XGB={roc_auc_score(y[va],oof_x[va]):.5f} CB={roc_auc_score(y[va],oof_c[va]):.5f}", flush=True)

print(f"\nLGBM OOF = {roc_auc_score(y, oof_l):.5f}", flush=True)
print(f"XGB  OOF = {roc_auc_score(y, oof_x):.5f}", flush=True)
print(f"CB   OOF = {roc_auc_score(y, oof_c):.5f}", flush=True)

# 保存 OOF + 测试预测, 供后续融合
np.save("cache/oof_lgb_pte.npy", oof_l); np.save("cache/pred_lgb_pte.npy", pred_l)
np.save("cache/oof_xgb_pte.npy", oof_x); np.save("cache/pred_xgb_pte.npy", pred_x)
np.save("cache/oof_cb_pte.npy", oof_c); np.save("cache/pred_cb_pte.npy", pred_c)
print("saved ensemble OOF/pred to cache/", flush=True)

for w1 in np.arange(0, 1.01, 0.1):
    for w2 in np.arange(0, 1.01-w1, 0.1):
        w3 = 1-w1-w2
        a = roc_auc_score(y, w1*rankdata(oof_l)/len(train) + w2*rankdata(oof_x)/len(train) + w3*rankdata(oof_c)/len(train))
        if w1 == 0.0 and w2 == 0.0:
            best = (a, (w1, w2, w3))
        elif a > best[0]:
            best = (a, (w1, w2, w3))
print(f"最优 rank-blend OOF = {best[0]:.5f} 权重={best[1]}", flush=True)
