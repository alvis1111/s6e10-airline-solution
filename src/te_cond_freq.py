"""S6E10 — 全套特征(基础TE + 条件TE + 数值分组TE + 频率编码) + GBDT 集成."""
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
NUMERIC = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
ENC_AUTO = dict(target_type='binary', smooth='auto', cv=5, shuffle=True, random_state=20261002)

def build(df):
    X = df.drop(columns=['id', 'satisfaction'], errors='ignore').copy()
    X['traveler_segment'] = X['Type of Travel'] + '|' + X['Customer Type'] + '|' + X['Class']
    X['arrival_missing'] = X['Arrival Delay in Minutes'].isna().astype('float32')
    for c in CATS + SERVICES + ['traveler_segment']:
        X[c] = X[c].astype(str).astype(object)
    return X

def conditional_features(df):
    seg = df['Type of Travel'].astype(str)+'|'+df['Customer Type'].astype(str)+'|'+df['Class'].astype(str)
    grp = df['Type of Travel'].astype(str)+'|'+df['Customer Type'].astype(str)
    z = pd.DataFrame(index=df.index)
    for c in SERVICES:
        z[c+'_segment'] = df[c].astype(str)+'|'+seg
    for c,d in [('Inflight wifi service','Online boarding'),('Inflight wifi service','Ease of Online booking'),
                ('Online boarding','Seat comfort'),('Seat comfort','Inflight entertainment'),
                ('Inflight entertainment','Cleanliness')]:
        z[c+'__'+d] = df[c].astype(str)+'|'+df[d].astype(str)+'|'+grp
    return z

def numeric_groups(df):
    return pd.DataFrame({f'{c}__{g}': df[c].astype(str)+'|'+df[g].astype(str)
        for c in ['Age','Flight Distance'] for g in ['Type of Travel','Customer Type','Class']}, index=df.index)

def te_transform(enc, Z, U, y, tr, va):
    et = enc.fit_transform(Z.iloc[tr], y[tr]).astype('float32')
    ev = enc.transform(Z.iloc[va]).astype('float32')
    ee = enc.transform(U).astype('float32')
    return et, ev, ee

def crossfit_frequency(Z, U, y, tr, va):
    z = Z.iloc[tr].reset_index(drop=True)
    fit_values = np.empty((len(tr), Z.shape[1]), dtype='float32')
    def transform(fit, apply):
        values = np.empty((len(apply), fit.shape[1]), dtype='float32')
        for j, col in enumerate(fit):
            density = fit[col].value_counts(dropna=False) / len(fit)
            values[:, j] = np.log1p(apply[col].map(density).fillna(0).to_numpy() * 1e6)
        return values
    cv = StratifiedKFold(5, shuffle=True, random_state=ENC_AUTO['random_state'])
    for fit, apply in cv.split(z, y[tr]):
        fit_values[apply] = transform(z.iloc[fit], z.iloc[apply])
    return fit_values, transform(z, Z.iloc[va]), transform(z, U)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train['satisfaction'].to_numpy(dtype=np.int64)
X, T = build(train), build(test)

# 全量预 fit label encoder (在 build 后的 X/T 上, 因 traveler_segment 是 build 里造的)
LES = {c: LabelEncoder().fit(pd.concat([X[c], T[c]]).astype(str)) for c in
       CATS + SERVICES + ['traveler_segment']}

# 预计算字符串帧
Z_base = train.drop(columns=['id','satisfaction']).astype(str)
U_base = test.drop(columns=['id']).astype(str)
Z_cond = conditional_features(train); U_cond = conditional_features(test)
Z_num = train[NUMERIC].astype(str); U_num = test[NUMERIC].astype(str)
G_tr = numeric_groups(train); G_te = numeric_groups(test)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = {'l': np.zeros(len(train)), 'x': np.zeros(len(train)), 'c': np.zeros(len(train))}
pred = {'l': np.zeros(len(test)), 'x': np.zeros(len(test)), 'c': np.zeros(len(test))}

for fold, (tr, va) in enumerate(skf.split(X, y)):
    Xt = X.iloc[tr].copy(); Xv = X.iloc[va].copy(); Xte = T.copy()
    # 数值 + label-encode 类别
    for f in [Xt, Xv, Xte]:
        for c in NUMERIC:
            f[c] = f[c].fillna(f[c].median() if f is Xt else Xt[c].median()).astype('float32')
        f['arrival_missing'] = f['arrival_missing'].astype('float32')
    for c in CATS + SERVICES + ['traveler_segment']:
        Xt[c] = LES[c].transform(Xt[c]); Xv[c] = LES[c].transform(Xv[c]); Xte[c] = LES[c].transform(Xte[c])

    # 1. 基础 TE (21列, smooth auto)
    enc = TargetEncoder(**ENC_AUTO)
    for f, vals in zip([Xt, Xv, Xte], te_transform(enc, Z_base, U_base, y, tr, va)):
        for i in range(vals.shape[1]): f[f'te_{i}'] = vals[:, i]
    # 2. 条件 TE (18列, smooth auto)
    enc = TargetEncoder(**ENC_AUTO)
    for f, vals in zip([Xt, Xv, Xte], te_transform(enc, Z_cond, U_cond, y, tr, va)):
        for i in range(vals.shape[1]): f[f'cond_te_{i}'] = vals[:, i]
    # 3. 数值分组 TE (6列, smooth 20)
    enc = TargetEncoder(**dict(ENC_AUTO, smooth=20))
    for f, vals in zip([Xt, Xv, Xte], te_transform(enc, G_tr, G_te, y, tr, va)):
        for i in range(vals.shape[1]): f[f'ng_te_{i}'] = vals[:, i]
    # 4. 频率编码 (数值+分组 = 10列)
    Zf = pd.concat([Z_num, G_tr], axis=1); Uf = pd.concat([U_num, G_te], axis=1)
    for f, vals in zip([Xt, Xv, Xte], crossfit_frequency(Zf, Uf, y, tr, va)):
        for i in range(vals.shape[1]): f[f'freq_{i}'] = vals[:, i]

    # 训练 3 模型
    m1 = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=63,
        min_child_samples=50, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
        reg_alpha=0.3, reg_lambda=1.0, n_estimators=8000, random_state=42, n_jobs=-1, verbosity=-1)
    m1.fit(Xt, y[tr], eval_X=Xv, eval_y=y[va], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof['l'][va] = m1.predict_proba(Xv)[:,1]; pred['l'] += m1.predict_proba(Xte)[:,1]/5

    m2 = xgb.XGBClassifier(objective="binary:logistic", eval_metric="auc", learning_rate=0.03,
        max_depth=8, min_child_weight=20, colsample_bytree=0.8, subsample=0.9,
        reg_alpha=0.1, reg_lambda=1.0, n_estimators=8000, tree_method="hist",
        random_state=42, n_jobs=-1, early_stopping_rounds=150)
    m2.fit(Xt, y[tr], eval_set=[(Xv, y[va])], verbose=False)
    oof['x'][va] = m2.predict_proba(Xv)[:,1]; pred['x'] += m2.predict_proba(Xte)[:,1]/5

    m3 = CatBoostClassifier(loss_function="Logloss", eval_metric="AUC", learning_rate=0.1,
        depth=6, l2_leaf_reg=3, iterations=3000, od_type="Iter", od_wait=100,
        verbose=False, thread_count=-1, random_seed=42)
    m3.fit(Xt, y[tr], eval_set=(Xv, y[va]), use_best_model=True)
    oof['c'][va] = m3.predict_proba(Xv)[:,1]; pred['c'] += m3.predict_proba(Xte)[:,1]/5
    print(f"fold {fold} done: LGB={roc_auc_score(y[va],oof['l'][va]):.5f} XGB={roc_auc_score(y[va],oof['x'][va]):.5f} CB={roc_auc_score(y[va],oof['c'][va]):.5f}", flush=True)

print(f"\nLGBM OOF = {roc_auc_score(y, oof['l']):.5f}", flush=True)
print(f"XGB  OOF = {roc_auc_score(y, oof['x']):.5f}", flush=True)
print(f"CB   OOF = {roc_auc_score(y, oof['c']):.5f}", flush=True)

def r(p): return rankdata(p)/len(p)
best = (0, None)
for w1 in np.arange(0, 1.01, 0.1):
    for w2 in np.arange(0, 1.01-w1, 0.1):
        w3 = 1-w1-w2
        a = roc_auc_score(y, w1*r(oof['l'])+w2*r(oof['x'])+w3*r(oof['c']))
        if a > best[0]: best = (a, (round(w1,1),round(w2,1),round(w3,1)))
print(f"最优 rank-blend OOF = {best[0]:.5f} 权重={best[1]}", flush=True)

for k in ['l','x','c']:
    np.save(f"cache/oof_cf_{k}.npy", oof[k]); np.save(f"cache/pred_cf_{k}.npy", pred[k])
print("saved", flush=True)
