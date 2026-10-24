"""l_oof_full — L 臂(raw 路由 + TE21/teacher 线性叶)全 5 折 OOF + 测试预测。

用于并入强融合测边际。TE21 为全 21 列 sklearn smooth='auto' 嵌套 OOF(外层 5 折, 内层 cv=5)。
"""
import time, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score

from repleafgbm.external import LightGBMExternalModel, RouterExtractionClassifier

DATA = r"C:\Users\GEM07\kaggle\s6e10-airline\data\playground-series-s6e10"
TEACHER = r"C:\Users\GEM07\Documents\Codex\2026-10-02\https-www-kaggle-com-competitions-playground\outputs\route_seed_experiment\production\teacher_predictions.npz"

N_EST = 600
N_LEAVES = 31
LR = 0.05

SERVICES = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
            'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
            'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness']
CATS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
NUMS = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']


def build_raw(train, test):
    X = train.drop(columns=['id', 'satisfaction']).copy()
    T = test.drop(columns=['id']).copy()
    for c in CATS:
        codes = {v: i for i, v in enumerate(sorted(X[c].dropna().unique()))}
        X[c] = X[c].map(codes)
        T[c] = T[c].map(codes).fillna(-1.0)
    med = {}
    for c in NUMS + SERVICES:
        X[c] = pd.to_numeric(X[c], errors='coerce')
        T[c] = pd.to_numeric(T[c], errors='coerce')
        med[c] = X[c].median()
        X[c] = X[c].fillna(med[c])
        T[c] = T[c].fillna(med[c])
    return X.astype(np.float32), T.astype(np.float32)


def gen_full_te(train, test, y, skf):
    Z = train.drop(columns=['id', 'satisfaction']).astype(str)
    U = test.drop(columns=['id']).astype(str)
    n_te = Z.shape[1]
    TE_COLS = [f'te_{i}' for i in range(n_te)]
    oof = np.zeros((len(train), n_te), dtype=np.float32)
    enc_params = dict(target_type='binary', smooth='auto', cv=5, shuffle=True, random_state=20261002)
    for tr, va in skf.split(train, y):
        enc = TargetEncoder(**enc_params)
        oof[tr] = enc.fit_transform(Z.iloc[tr], y[tr])
        oof[va] = enc.transform(Z.iloc[va])
    enc = TargetEncoder(**enc_params)
    enc.fit(Z, y)
    test_te = enc.transform(U).astype(np.float32)
    return oof, test_te, TE_COLS


def main():
    t0 = time.time()
    train = pd.read_csv(f"{DATA}/train.csv")
    test = pd.read_csv(f"{DATA}/test.csv")
    y = train['satisfaction'].to_numpy(np.int64)
    teacher = np.load(TEACHER)
    t_train = teacher['train'].astype(np.float32)
    t_test = teacher['test'].astype(np.float32)
    print(f"数据加载 ({time.time()-t0:.1f}s)", flush=True)

    Xraw_train, Xraw_test = build_raw(train, test)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    te_oof, te_test, TE_COLS = gen_full_te(train, test, y, skf)
    print(f"TE21 生成完成 ({time.time()-t0:.1f}s)", flush=True)

    Xfull_train = np.hstack([Xraw_train.to_numpy(), te_oof, t_train[:, None]])
    Xfull_test = np.hstack([Xraw_test.to_numpy(), te_test, t_test[:, None]])
    cols = list(Xraw_train.columns) + TE_COLS + ['teacher']
    Xfull_train = pd.DataFrame(Xfull_train, columns=cols)
    Xfull_test = pd.DataFrame(Xfull_test, columns=cols)

    oof = np.zeros(len(train), dtype=np.float64)
    pred = np.zeros(len(test), dtype=np.float64)

    for fold, (tr, va) in enumerate(skf.split(Xfull_train, y)):
        t = time.time()
        base = LightGBMExternalModel(task='binary', random_state=42,
                                     n_estimators=N_EST, num_leaves=N_LEAVES,
                                     learning_rate=LR, min_child_samples=20,
                                     n_jobs=-1, verbosity=-1)
        clf = RouterExtractionClassifier(
            base=base, leaf_model='embedded_linear',
            leaf_only_features=TE_COLS + ['teacher'],
            leaf_experts=[
                {'name': 'TE21', 'encoder': 'identity', 'columns': TE_COLS},
                {'name': 'O', 'encoder': 'identity', 'columns': ['teacher']},
            ],
            random_state=42,
        )
        clf.fit(Xfull_train.iloc[tr], y[tr])
        oof[va] = clf.predict_proba(Xfull_train.iloc[va])[:, 1]
        pred += clf.predict_proba(Xfull_test)[:, 1] / 5
        auc = roc_auc_score(y[va], oof[va])
        print(f"fold {fold}: L val_auc={auc:.5f}  ({time.time()-t:.0f}s)", flush=True)

    full_auc = roc_auc_score(y, oof)
    print(f"\nL 全 5 折 OOF = {full_auc:.5f}", flush=True)
    os.makedirs("cache", exist_ok=True)
    np.save("cache/oof_L_te21.npy", oof)
    np.save("cache/pred_L_te21.npy", pred)
    print(f"已保存 cache/oof_L_te21.npy + cache/pred_L_te21.npy  总耗时 {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
