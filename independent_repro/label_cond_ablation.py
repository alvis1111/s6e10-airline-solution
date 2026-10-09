"""标签条件辅助任务（Wi-Fi）正确消融：53 列 vs 54 列 LR。

1. 外层 fold 0（5 折 seed 33）不变。
2. 外层训练集 559,708 行上 3 折内层交叉预测，生成每行的 log-ratio（代入 label=0/1，不传自身真实满意度）。
3. 外层验证行的 log-ratio 复用已核验缓存 cache/label_cond_fold0.npz。
4. 53 列基线 vs 54 列候选 LR（同 C=1.0），比较 fold 0 边际。
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("data/playground-series-s6e10/train.csv")
y = train["satisfaction"].to_numpy(np.int64)

RATINGS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
           "Gate location", "Food and drink", "Online boarding", "Seat comfort",
           "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
           "Checkin service", "Cleanliness"]
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
WIFI = "Inflight wifi service"

base = train.drop(columns=["id", "satisfaction"]).copy()
base["Arrival Delay in Minutes"] = base["Arrival Delay in Minutes"].fillna(0.0)
for c in CATS:
    base[c] = base[c].astype("category")
wifi = train[WIFI].to_numpy(np.int64)
label = y


def make_X(df, lab):
    X = df.drop(columns=[WIFI]).copy()
    X["__label__"] = lab
    return X


def train_predictor(rows):
    m = lgb.LGBMClassifier(objective="multiclass", num_class=6, n_estimators=200, learning_rate=0.1,
                           num_leaves=31, n_jobs=-1, random_state=0, verbosity=-1)
    m.fit(make_X(base.iloc[rows], label[rows]), wifi[rows])
    return m


def logratio_of(m, rows):
    p1 = m.predict_proba(make_X(base.iloc[rows], np.ones(len(rows), dtype=np.int64)))
    p0 = m.predict_proba(make_X(base.iloc[rows], np.zeros(len(rows), dtype=np.int64)))
    return np.array([np.log(p1[i, wifi[rows][i]] + 1e-9) - np.log(p0[i, wifi[rows][i]] + 1e-9)
                     for i in range(len(rows))])


logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

folds = list(StratifiedKFold(5, shuffle=True, random_state=33).split(base, y))
ia, iv = folds[0]
print(f"fold 0: train {len(ia)} / val {len(iv)}", flush=True)

# 外层训练集 3 折内层交叉预测，生成训练行 log-ratio
logratio_ia = np.zeros(len(ia))
inner = StratifiedKFold(3, shuffle=True, random_state=0).split(base.iloc[ia], y[ia])
for k, (i_tr, i_va) in enumerate(inner):
    m = train_predictor(ia[i_tr])
    logratio_ia[i_va] = logratio_of(m, ia[i_va])
    print(f"  内层 fold {k}: 完成，训练 {len(i_tr)} 行", flush=True)

# 外层验证行 log-ratio 复用缓存
d = np.load("cache/label_cond_fold0.npz", allow_pickle=False)
assert np.array_equal(d["val_id"], train["id"].iloc[iv].to_numpy())
logratio_iv = d["logratio"]
print("外层验证 log-ratio 已复用缓存", flush=True)

# 53 列基线 vs 54 列候选 LR
MEM = r"C:/Users/GEM07/.cache/kagglehub/notebooks/goodpjw2008/s6e10-tabpfn-route-categories-lb-0-96160/output/versions/1"
mo = pd.read_csv(MEM + "/oof_members.csv").set_index("id").reindex(train["id"])
z = np.load("pulled_busy53/oof_ours.npz", allow_pickle=False)
X53 = np.column_stack([logit(mo[c].to_numpy()) for c in mo.columns] + [logit(z[n]) for n in ["lgbm", "lgbm_xt", "cat", "xgb", "realmlp"]])

X54 = np.column_stack([X53, np.zeros(len(train))])
X54[ia, -1] = logratio_ia
X54[iv, -1] = logratio_iv

m53 = LogisticRegression(max_iter=2000, C=1.0).fit(X53[ia], y[ia])
m54 = LogisticRegression(max_iter=2000, C=1.0).fit(X54[ia], y[ia])
a53 = roc_auc_score(y[iv], m53.decision_function(X53[iv]))
a54 = roc_auc_score(y[iv], m54.decision_function(X54[iv]))
print(f"\n53 列 LR fold0 AUC = {a53:.6f}", flush=True)
print(f"54 列 LR fold0 AUC = {a54:.6f}  Δ {a54-a53:+.6f}", flush=True)
print(f"新特征系数 = {m54.coef_[0][-1]:+.4f}（相对 53 系数平均量级 {np.abs(m53.coef_[0]).mean():.4f}）", flush=True)
