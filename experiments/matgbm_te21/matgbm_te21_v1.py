"""matgbm_te21_v1 — 全 21 列 TE + teacher 作为线性叶空间, 验证线性叶机制。

空间 = sklearn TargetEncoder(smooth='auto', cv=5) 对全 21 列做 fold-safe OOF(嵌套:
外层 StratifiedKFold(42), 内层 cv=5)+ teacher 1 列, 共 22 列。

三臂(外层 fold 0):
  B = raw + 常数叶
  R = raw + TE21 + teacher + 常数叶
  L = raw 路由 + TE21/teacher 进叶 + embedded_linear 线性叶
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
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
    """全 21 列 fold-safe OOF TE(sklearn TargetEncoder, smooth auto, cv=5, 嵌套)。"""
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
    print(f"数据加载 train={len(train)} test={len(test)} ({time.time()-t0:.1f}s)", flush=True)

    Xraw_train, Xraw_test = build_raw(train, test)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    te_oof, te_test, TE_COLS = gen_full_te(train, test, y, skf)
    print(f"全21列TE生成完成, {len(TE_COLS)} 列 ({time.time()-t0:.1f}s)", flush=True)

    Xfull_train = np.hstack([Xraw_train.to_numpy(), te_oof, t_train[:, None]])
    Xfull_test = np.hstack([Xraw_test.to_numpy(), te_test, t_test[:, None]])
    cols = list(Xraw_train.columns) + TE_COLS + ['teacher']
    Xfull_train = pd.DataFrame(Xfull_train, columns=cols)
    Xfull_test = pd.DataFrame(Xfull_test, columns=cols)
    print(f"特征矩阵: {Xfull_train.shape}", flush=True)

    tr, va = list(skf.split(Xfull_train, y))[1]  # fold 1 确认
    print(f"fold-1: train={len(tr)} valid={len(va)}", flush=True)

    Xraw_tr, Xraw_va = Xraw_train.iloc[tr], Xraw_train.iloc[va]
    Xfull_tr, Xfull_va = Xfull_train.iloc[tr], Xfull_train.iloc[va]
    ytr, yva = y[tr], y[va]

    lgb_params = dict(objective='binary', metric='auc', learning_rate=LR,
                      num_leaves=N_LEAVES, min_child_samples=20, n_estimators=N_EST,
                      random_state=42, n_jobs=-1, verbosity=-1)

    def fit_B(Xtr, ytr, Xva):
        m = lgb.LGBMClassifier(**lgb_params); m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1]

    def fit_R(Xtr, ytr, Xva):
        m = lgb.LGBMClassifier(**lgb_params); m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1]

    def fit_L(Xtr, ytr, Xva):
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
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xva)[:, 1]

    print("--- 三臂结果 (fold 0, 空间=全21列TE+teacher) ---", flush=True)
    import os
    os.makedirs("cache", exist_ok=True)
    preds = {}
    for name, fit_fn, Xtr, Xva in [('B raw+常数叶', fit_B, Xraw_tr, Xraw_va),
                                    ('R raw+TE21+teacher+常数叶', fit_R, Xfull_tr, Xfull_va),
                                    ('L raw路由+TE21/teacher线性叶', fit_L, Xfull_tr, Xfull_va)]:
        t = time.time()
        pred = fit_fn(Xtr, ytr, Xva)
        a = roc_auc_score(yva, pred)
        preds[name.split()[0]] = pred
        print(f"  [{name}] AUC={a:.5f}  耗时={time.time()-t:.1f}s", flush=True)
    auc_B = roc_auc_score(yva, preds['B'])
    auc_R = roc_auc_score(yva, preds['R'])
    auc_L = roc_auc_score(yva, preds['L'])
    fusion = 0.5 * preds['R'] + 0.5 * preds['L']
    auc_fuse = roc_auc_score(yva, fusion)
    np.save("cache/te21_fold1_pred_R.npy", preds['R'])
    np.save("cache/te21_fold1_pred_L.npy", preds['L'])
    np.save("cache/te21_fold1_pred_fusion.npy", fusion)
    # 相关性(检验 L 是否提供互补信号)
    corr = float(np.corrcoef(preds['R'], preds['L'])[0, 1])
    print(f"\nAUC(R)={auc_R:.5f}  AUC(L)={auc_L:.5f}  AUC(0.5R+0.5L)={auc_fuse:.5f}  corr(R,L)={corr:.6f}", flush=True)
    print(f"Δ(fusion - R) = {auc_fuse-auc_R:+.5f}", flush=True)
    print(f"Δ(fusion - L) = {auc_fuse-auc_L:+.5f}", flush=True)
    print(f"Δ(R-B) 加特征 = {auc_R-auc_B:+.5f}", flush=True)
    print(f"Δ(L-B) 线性叶总增益 = {auc_L-auc_B:+.5f}", flush=True)
    print(f"Δ(L-R) 线性叶机制 = {auc_L-auc_R:+.5f}", flush=True)
    print(f"总耗时 {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
