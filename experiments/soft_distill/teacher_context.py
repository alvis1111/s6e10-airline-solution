"""teacher_context — context 腿(条件 TE + 频率 LGBM)的教师 OOF(fold 1-9 嵌套 5 折)。"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score

DATA = "data/playground-series-s6e10"
CATS = ['Gender','Customer Type','Type of Travel','Class']
SERVICES = ['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location',
    'Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service',
    'Baggage handling','Checkin service','Cleanliness']
NUMERIC = ['Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes']
ENC_AUTO = dict(target_type='binary', smooth='auto', cv=5, shuffle=True, random_state=20261002)

def build(df):
    X = df.drop(columns=['id','satisfaction'], errors='ignore').copy()
    X['traveler_segment'] = X['Type of Travel']+'|'+X['Customer Type']+'|'+X['Class']
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

def te_transform(enc, Z, U, y, tr, va):
    return (enc.fit_transform(Z.iloc[tr], y[tr]).astype('float32'),
            enc.transform(Z.iloc[va]).astype('float32'))

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
    return fit_values, transform(z, Z.iloc[va])

train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv")
y = train['satisfaction'].to_numpy(dtype=np.int64)
X, T = build(train), build(test)
LES = {c: LabelEncoder().fit(pd.concat([X[c], T[c]]).astype(str)) for c in CATS+SERVICES+['traveler_segment']}
Z_base = train.drop(columns=['id','satisfaction']).astype(str)
Z_cond = conditional_features(train)
Z_num = train[NUMERIC].astype(str)
G_tr = numeric_groups(train)

outer = list(StratifiedKFold(10, shuffle=True, random_state=42).split(X, y))
_, iv = outer[1]
ia = np.setdiff1d(np.arange(len(X)), iv)
teacher_oof = np.full(len(X), np.nan)
y_s = pd.Series(y)

inner = StratifiedKFold(5, shuffle=True, random_state=42)
for fold, (tr, va) in enumerate(inner.split(ia, y[ia])):
    itr, iva = ia[tr], ia[va]
    Xt = X.iloc[itr].copy(); Xv = X.iloc[iva].copy()
    for f in [Xt, Xv]:
        for c in NUMERIC:
            f[c] = f[c].fillna(Xt[c].median()).astype('float32')
        f['arrival_missing'] = f['arrival_missing'].astype('float32')
    for c in CATS+SERVICES+['traveler_segment']:
        Xt[c] = LES[c].transform(Xt[c]); Xv[c] = LES[c].transform(Xv[c])
    # 基础 TE
    enc = TargetEncoder(**ENC_AUTO)
    et, ev = te_transform(enc, Z_base, Z_base, y_s, itr, iva)
    for i in range(et.shape[1]): Xt[f'te_{i}'] = et[:, i]; Xv[f'te_{i}'] = ev[:, i]
    # 条件 TE
    enc = TargetEncoder(**ENC_AUTO)
    et, ev = te_transform(enc, Z_cond, Z_cond, y_s, itr, iva)
    for i in range(et.shape[1]): Xt[f'cond_te_{i}'] = et[:, i]; Xv[f'cond_te_{i}'] = ev[:, i]
    # 数值分组 TE
    enc = TargetEncoder(**dict(ENC_AUTO, smooth=20))
    et, ev = te_transform(enc, G_tr, G_tr, y_s, itr, iva)
    for i in range(et.shape[1]): Xt[f'ng_te_{i}'] = et[:, i]; Xv[f'ng_te_{i}'] = ev[:, i]
    # 频率
    Zf = pd.concat([Z_num, G_tr], axis=1)
    fv, ff = crossfit_frequency(Zf, Zf, y_s, itr, iva)
    for i in range(fv.shape[1]): Xt[f'freq_{i}'] = fv[:, i]; Xv[f'freq_{i}'] = ff[:, i]

    m1 = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=63,
        min_child_samples=50, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
        reg_alpha=0.3, reg_lambda=1.0, n_estimators=8000, random_state=42, n_jobs=-1, verbosity=-1)
    m1.fit(Xt, y[itr], eval_X=Xv, eval_y=y[iva], callbacks=[lgb.early_stopping(150, verbose=False)])
    teacher_oof[iva] = m1.predict_proba(Xv)[:,1]
    print(f"inner fold {fold}: context LGB AUC={roc_auc_score(y[iva], teacher_oof[iva]):.5f}", flush=True)

np.save("cache/teacher_context_oof.npy", teacher_oof)
print(f"\ncontext 腿教师 OOF 完成, 覆盖 {np.isfinite(teacher_oof).sum()} 行", flush=True)
