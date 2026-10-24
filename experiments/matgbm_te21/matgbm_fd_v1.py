"""FD_multiscale_v1 — 验证线性叶机制本身是否有增益(不追求复现作者 +0.0014)。

三臂(外层 StratifiedKFold(5, shuffle=True, random_state=42),取 fold 0 对比):
  B = raw 特征 + 常数叶(标准 LightGBM)
  R = raw + FD_multiscale_v1(8列) + teacher + 常数叶(测"加特征")
  L = raw 路由 + FD/teacher 进叶 + embedded_linear 线性叶(测"线性叶机制")

FD_multiscale_v1: 单 key = Flight Distance, 4 种分组(exact / 四舍五入到十位 /
四舍五入到百位 / 50 分位箱),每组输出 平滑目标均值(alpha=5) + log1p(count),共 8 列。
外层训练集内部 5 折交叉拟合(OOF),外层验证用外层训练整体 fit 的冻结表,标签完全隔离。

SMOKE=True 用小样本+少树跑通并计时;SMOKE=False 跑正式 fold 0。
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from repleafgbm.external import LightGBMExternalModel, RouterExtractionClassifier
from repleafgbm.encoders.target_encoding import TargetEncodingEncoder

DATA = r"C:\Users\GEM07\kaggle\s6e10-airline\data\playground-series-s6e10"
TEACHER = r"C:\Users\GEM07\Documents\Codex\2026-10-02\https-www-kaggle-com-competitions-playground\outputs\route_seed_experiment\production\teacher_predictions.npz"

SMOKE = False
N_SMOKE = 30000          # 冒烟时从 fold-0 训练集抽样
N_EST = 100 if SMOKE else 600
N_LEAVES = 31
LR = 0.05

SERVICES = ['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
            'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
            'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Cleanliness']
CATS = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
NUMS = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
FD = 'Flight Distance'
FD8_COLS = ['fd_te_exact', 'fd_cnt_exact', 'fd_te_r10', 'fd_cnt_r10',
            'fd_te_r100', 'fd_cnt_r100', 'fd_te_bin50', 'fd_cnt_bin50']


def build_raw(train, test):
    """数值化原始特征: 类别列序数编码, 数值列中位数填充(中位数只在 train 上 fit)。"""
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
    X = X.astype(np.float32)
    T = T.astype(np.float32)
    return X, T


def gen_fd_multiscale(fd_train, y, fd_test, skf):
    """嵌套 OOF 生成 FD_multiscale_v1 8 列(train 行 OOF, test 行 full-fit)。"""
    enc_kwargs = dict(columns=[0], scales=["exact", -1, -2, "bin:50"],
                      alpha=5, n_folds=5, add_count=True, random_state=42)
    oof = np.zeros((fd_train.shape[0], 8), dtype=np.float64)
    for tr, va in skf.split(fd_train, y):
        enc = TargetEncodingEncoder(**enc_kwargs)
        oof[tr] = enc.fit_transform_train(fd_train[tr], y[tr])
        oof[va] = enc.transform(fd_train[va])
    enc = TargetEncodingEncoder(**enc_kwargs)
    enc.fit(fd_train, y)
    test_te = enc.transform(fd_test)
    return oof, test_te


def run_arm(name, fit_fn, Xtr, ytr, Xva, yva):
    t0 = time.time()
    pred = fit_fn(Xtr, ytr, Xva)
    auc = roc_auc_score(yva, pred)
    print(f"  [{name}] AUC={auc:.5f}  耗时={time.time()-t0:.1f}s", flush=True)
    return auc


def main():
    t0 = time.time()
    print(f"SMOKE={SMOKE}  N_EST={N_EST}", flush=True)
    train = pd.read_csv(f"{DATA}/train.csv")
    test = pd.read_csv(f"{DATA}/test.csv")
    y = train['satisfaction'].to_numpy(np.int64)
    teacher = np.load(TEACHER)
    t_train = teacher['train'].astype(np.float32)
    t_test = teacher['test'].astype(np.float32)
    print(f"数据加载完成 train={len(train)} test={len(test)}  ({time.time()-t0:.1f}s)", flush=True)

    Xraw_train, Xraw_test = build_raw(train, test)
    fd_train = Xraw_train[[FD]].to_numpy(np.float64)
    fd_test = Xraw_test[[FD]].to_numpy(np.float64)

    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    fd_oof, fd_test_te = gen_fd_multiscale(fd_train, y, fd_test, skf)
    print(f"FD_multiscale_v1 生成完成 ({time.time()-t0:.1f}s)", flush=True)

    # 完整特征矩阵: raw21 + fd8 + teacher1 = 30 列
    Xfull_train = np.hstack([Xraw_train.to_numpy(), fd_oof, t_train[:, None]])
    Xfull_test = np.hstack([Xraw_test.to_numpy(), fd_test_te, t_test[:, None]])
    cols = list(Xraw_train.columns) + FD8_COLS + ['teacher']
    Xfull_train = pd.DataFrame(Xfull_train, columns=cols)
    Xfull_test = pd.DataFrame(Xfull_test, columns=cols)
    print(f"特征矩阵: {Xfull_train.shape} (raw21 + fd8 + teacher)", flush=True)

    # 外层 fold 0
    tr, va = next(iter(skf.split(Xfull_train, y)))
    if SMOKE:
        rng = np.random.default_rng(0)
        tr = rng.choice(tr, size=N_SMOKE, replace=False)
    print(f"fold-0: train={len(tr)} valid={len(va)}", flush=True)

    Xraw_tr, Xraw_va = Xraw_train.iloc[tr], Xraw_train.iloc[va]
    Xfull_tr, Xfull_va = Xfull_train.iloc[tr], Xfull_train.iloc[va]
    ytr, yva = y[tr], y[va]

    lgb_params = dict(objective='binary', metric='auc', learning_rate=LR,
                      num_leaves=N_LEAVES, min_child_samples=20, n_estimators=N_EST,
                      random_state=42, n_jobs=-1, verbosity=-1)

    # B: raw + 常数叶
    def fit_B(Xtr, ytr, Xva):
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1]

    # R: raw + fd + teacher + 常数叶
    def fit_R(Xtr, ytr, Xva):
        m = lgb.LGBMClassifier(**lgb_params)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xva)[:, 1]

    # L: raw 路由 + fd/teacher 进叶 + 线性叶
    def fit_L(Xtr, ytr, Xva):
        base = LightGBMExternalModel(task='binary', random_state=42,
                                     n_estimators=N_EST, num_leaves=N_LEAVES,
                                     learning_rate=LR, min_child_samples=20,
                                     n_jobs=-1, verbosity=-1)
        clf = RouterExtractionClassifier(
            base=base, leaf_model='embedded_linear',
            leaf_only_features=FD8_COLS + ['teacher'],
            leaf_experts=[
                {'name': 'FD8', 'encoder': 'identity', 'columns': FD8_COLS},
                {'name': 'O', 'encoder': 'identity', 'columns': ['teacher']},
            ],
            random_state=42,
        )
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xva)[:, 1]

    print("--- 三臂结果 (fold 0) ---", flush=True)
    auc_B = run_arm('B raw+常数叶', fit_B, Xraw_tr, ytr, Xraw_va, yva)
    auc_R = run_arm('R raw+fd+teacher+常数叶', fit_R, Xfull_tr, ytr, Xfull_va, yva)
    auc_L = run_arm('L raw路由+fd/teacher线性叶', fit_L, Xfull_tr, ytr, Xfull_va, yva)
    print(f"\nΔ(R-B) 加特征 = {auc_R-auc_B:+.5f}", flush=True)
    print(f"Δ(L-B) 线性叶总增益 = {auc_L-auc_B:+.5f}", flush=True)
    print(f"Δ(L-R) 线性叶机制(相对加特征) = {auc_L-auc_R:+.5f}", flush=True)
    print(f"总耗时 {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
