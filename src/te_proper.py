"""S6E10 — 复现并行会话的核心: sklearn TargetEncoder(smooth auto, cv=5, 全21列) + LGBM."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

DATA = "data/playground-series-s6e10"
CATS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
SERVICES = ['Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding',
    'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Cleanliness']
ENC_PARAMS = dict(target_type='binary', smooth='auto', cv=5, shuffle=True, random_state=20261002)

def build(df):
    X = df.drop(columns=['id', 'satisfaction'], errors='ignore').copy()
    X['traveler_segment'] = X['Type of Travel'] + '|' + X['Customer Type'] + '|' + X['Class']
    X['arrival_missing'] = X['Arrival Delay in Minutes'].isna().astype('float32')
    for c in CATS + SERVICES + ['traveler_segment']:
        X[c] = X[c].astype(str).astype(object)
    return X

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train['satisfaction'].to_numpy(dtype=np.int64)
X, T = build(train), build(test)

def fold_matrices(tr, va):
    Xt, Xv, Xte = X.iloc[tr].copy(), X.iloc[va].copy(), T.copy()
    for c in X:
        if c in CATS + SERVICES + ['traveler_segment']:
            dtype = pd.CategoricalDtype(sorted(Xt[c].unique()))
            for f in [Xt, Xv, Xte]:
                f[c] = f[c].astype(dtype)
        else:
            median = Xt[c].median()
            for f in [Xt, Xv, Xte]:
                f[c] = f[c].fillna(median).astype('float32')
    Z = train.drop(columns=['id', 'satisfaction']).astype(str)
    U = test.drop(columns=['id']).astype(str)
    enc = TargetEncoder(**ENC_PARAMS)
    et = enc.fit_transform(Z.iloc[tr], y[tr]).astype('float32')
    ev = enc.transform(Z.iloc[va]).astype('float32')
    ee = enc.transform(U).astype('float32')
    for f, encoded in [(Xt, et), (Xv, ev), (Xte, ee)]:
        for i in range(encoded.shape[1]):
            f[f'te_{i}'] = encoded[:, i]
    return Xt, Xv, Xte

skf = StratifiedKFold(5, shuffle=True, random_state=42)
lgb_params = dict(objective="binary", metric="auc", learning_rate=0.03, num_leaves=127,
                  min_child_samples=50, colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                  reg_alpha=0.3, reg_lambda=1.0, n_estimators=8000, random_state=42, n_jobs=-1, verbosity=-1)

oof = np.zeros(len(train)); pred = np.zeros(len(test))
for fold, (tr, va) in enumerate(skf.split(X, y)):
    Xt, Xv, Xte = fold_matrices(tr, va)
    m = lgb.LGBMClassifier(**lgb_params)
    m.fit(Xt, y[tr], eval_X=Xv, eval_y=y[va], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[va] = m.predict_proba(Xv)[:, 1]
    pred += m.predict_proba(Xte)[:, 1] / 5
    print(f"fold {fold}: best_iter={m.best_iteration_} val_auc={roc_auc_score(y[va], oof[va]):.5f}", flush=True)

print(f"\nproper-TE LGBM OOF = {roc_auc_score(y, oof):.5f}", flush=True)
np.save("cache/oof_te_proper_lgb.npy", oof)
np.save("cache/pred_te_proper_lgb.npy", pred)
