"""S6E10 — 验证「服务评分 × 年龄/航程」交互 TE 是否有效 (单 LGBM 快速验证)."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score

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
    for c in SERVICES: z[c+'_segment'] = df[c].astype(str)+'|'+seg
    for c,d in [('Inflight wifi service','Online boarding'),('Inflight wifi service','Ease of Online booking'),
                ('Online boarding','Seat comfort'),('Seat comfort','Inflight entertainment'),
                ('Inflight entertainment','Cleanliness')]:
        z[c+'__'+d] = df[c].astype(str)+'|'+df[d].astype(str)+'|'+grp
    return z

def numeric_groups(df):
    return pd.DataFrame({f'{c}__{g}': df[c].astype(str)+'|'+df[g].astype(str)
        for c in ['Age','Flight Distance'] for g in ['Type of Travel','Customer Type','Class']}, index=df.index)

def svc_cont(df):
    """服务评分 × 年龄桶 / 航程桶 的组合列."""
    age_b = pd.cut(df['Age'], bins=[0,20,30,40,50,60,70,100], labels=False).astype(str)
    dist_b = pd.cut(df['Flight Distance'], bins=[0,300,600,1000,1500,2500,5000,np.inf], labels=False).astype(str)
    z = pd.DataFrame(index=df.index)
    for c in SERVICES:
        z[c+'__age'] = df[c].astype(str)+'|'+age_b
        z[c+'__dist'] = df[c].astype(str)+'|'+dist_b
    return z

def te_transform(enc, Z, U, y, tr, va):
    return (enc.fit_transform(Z.iloc[tr], y[tr]).astype('float32'),
            enc.transform(Z.iloc[va]).astype('float32'),
            enc.transform(U).astype('float32'))

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

train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv")
y = train['satisfaction'].to_numpy(dtype=np.int64)
X, T = build(train), build(test)

LES = {c: LabelEncoder().fit(pd.concat([X[c], T[c]]).astype(str)) for c in CATS + SERVICES + ['traveler_segment']}
Z_base = train.drop(columns=['id','satisfaction']).astype(str); U_base = test.drop(columns=['id']).astype(str)
Z_cond = conditional_features(train); U_cond = conditional_features(test)
Z_num = train[NUMERIC].astype(str); U_num = test[NUMERIC].astype(str)
G_tr = numeric_groups(train); G_te = numeric_groups(test)
S_tr = svc_cont(train); S_te = svc_cont(test)

def run(use_svc_cont):
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.zeros(len(train))
    for tr, va in skf.split(X, y):
        Xt = X.iloc[tr].copy(); Xv = X.iloc[va].copy(); Xte = T.copy()
        for f in [Xt, Xv, Xte]:
            for c in NUMERIC:
                f[c] = f[c].fillna(f[c].median() if f is Xt else Xt[c].median()).astype('float32')
            f['arrival_missing'] = f['arrival_missing'].astype('float32')
        for c in CATS + SERVICES + ['traveler_segment']:
            Xt[c] = LES[c].transform(Xt[c]); Xv[c] = LES[c].transform(Xv[c]); Xte[c] = LES[c].transform(Xte[c])
        # 1 基础 TE
        enc = TargetEncoder(**ENC_AUTO)
        for f, vals in zip([Xt,Xv,Xte], te_transform(enc, Z_base, U_base, y, tr, va)):
            for i in range(vals.shape[1]): f[f'te_{i}'] = vals[:,i]
        # 2 条件 TE
        enc = TargetEncoder(**ENC_AUTO)
        for f, vals in zip([Xt,Xv,Xte], te_transform(enc, Z_cond, U_cond, y, tr, va)):
            for i in range(vals.shape[1]): f[f'cond_te_{i}'] = vals[:,i]
        # 3 数值分组 TE
        enc = TargetEncoder(**dict(ENC_AUTO, smooth=20))
        for f, vals in zip([Xt,Xv,Xte], te_transform(enc, G_tr, G_te, y, tr, va)):
            for i in range(vals.shape[1]): f[f'ng_te_{i}'] = vals[:,i]
        # 4 频率
        Zf = pd.concat([Z_num, G_tr], axis=1); Uf = pd.concat([U_num, G_te], axis=1)
        for f, vals in zip([Xt,Xv,Xte], crossfit_frequency(Zf, Uf, y, tr, va)):
            for i in range(vals.shape[1]): f[f'freq_{i}'] = vals[:,i]
        # 5 (新) 服务评分×年龄/航程 TE
        if use_svc_cont:
            enc = TargetEncoder(**ENC_AUTO)
            for f, vals in zip([Xt,Xv,Xte], te_transform(enc, S_tr, S_te, y, tr, va)):
                for i in range(vals.shape[1]): f[f'sc_te_{i}'] = vals[:,i]
        m = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=63,
            min_child_samples=50, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
            reg_alpha=0.3, reg_lambda=1.0, n_estimators=8000, random_state=42, n_jobs=-1, verbosity=-1)
        m.fit(Xt, y[tr], eval_X=Xv, eval_y=y[va], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = m.predict_proba(Xv)[:,1]
    return roc_auc_score(y, oof)

print("=== 服务评分×年龄/航程 交互 TE 验证 ===", flush=True)
print(f"A. 全套特征(不含新交互)     OOF = {run(False):.5f}", flush=True)
print(f"B. +服务评分×年龄/航程交互   OOF = {run(True):.5f}", flush=True)
