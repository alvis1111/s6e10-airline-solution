"""S6E10 — TabM k=32 (升级) + 保存 OOF/pred."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
from pytabkit import TabM_D_Classifier
import torch

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

MODEL_PARAMS = dict(device='cuda:0', random_state=3407, n_cv=1, n_refit=0,
    n_threads=6, verbosity=0, val_metric_name='1-auc_ovr', n_epochs=128, batch_size=1024,
    tabm_k=32, num_emb_type='pwl', num_emb_n_bins=32, d_embedding=16,
    d_block=256, n_blocks=2, dropout=.1, patience=16, allow_amp=True,
    compile_model=False, share_training_batches=False)

torch.set_num_threads(6)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(train)); pred = np.zeros(len(test))
for fold, (tr, va) in enumerate(skf.split(X, y)):
    Xt, Xv, Xte = fold_matrices(tr, va)
    model = TabM_D_Classifier(**MODEL_PARAMS)
    model.fit(Xt, y[tr], X_val=Xv, y_val=y[va])
    pv = model.predict_proba(Xv)[:, 1]
    pt = model.predict_proba(Xte)[:, 1]
    oof[va] = pv; pred += pt / 5
    print(f"fold {fold}: AUC={roc_auc_score(y[va], pv):.5f}", flush=True)
    del model; torch.cuda.empty_cache()

print(f"\nTabM k=32 OOF = {roc_auc_score(y, oof):.5f}", flush=True)
np.save("cache/oof_tabm_k32.npy", oof)
np.save("cache/pred_tabm_k32.npy", pred)
