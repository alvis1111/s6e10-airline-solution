"""标签条件辅助任务（Wi-Fi 列）首筛：fold 0。

训练一个「用其他 20 列 + 满意度标签预测 Wi-Fi 评分」的多分类器（折内训练）。
对验证行分别代入 label=1/0 两种假设，取「实际 Wi-Fi 评分」的对数概率比作为新特征，
10% logit 加进 Busy53 看边际。验证行不使用其真实满意度。
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

train = pd.read_csv("data/playground-series-s6e10/train.csv")
y = train["satisfaction"].to_numpy(np.int64)

RATINGS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
           "Gate location", "Food and drink", "Online boarding", "Seat comfort",
           "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
           "Checkin service", "Cleanliness"]
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
WIFI = "Inflight wifi service"

# 特征 = 其他 20 列 + 满意度标签
base = train.drop(columns=["id", "satisfaction"]).copy()
base["Arrival Delay in Minutes"] = base["Arrival Delay in Minutes"].fillna(0.0)
for c in CATS:
    base[c] = base[c].astype("category")

wifi = train[WIFI].to_numpy(np.int64)
label = y  # 满意度 0/1


def make_X(df, lab):
    X = df.drop(columns=[WIFI]).copy()  # 其他 20 列
    X["__label__"] = lab
    return X


logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

folds = list(StratifiedKFold(5, shuffle=True, random_state=33).split(base, y))
ia, iv = folds[0]
print(f"fold 0: train {len(ia)} / val {len(iv)}", flush=True)

# 折内训练 M：预测 Wi-Fi（0-5，6 类）
X_tr = make_X(base.iloc[ia], label[ia])
m = lgb.LGBMClassifier(objective="multiclass", num_class=6, n_estimators=200, learning_rate=0.1,
                       num_leaves=31, n_jobs=-1, random_state=0, verbosity=-1)
m.fit(X_tr, wifi[ia])
print("Wi-Fi 预测器训练完成", flush=True)

# 验证行分别代入两种假设
p1 = m.predict_proba(make_X(base.iloc[iv], np.ones(len(iv), dtype=np.int64)))  # label=1
p0 = m.predict_proba(make_X(base.iloc[iv], np.zeros(len(iv), dtype=np.int64)))  # label=0
logratio = np.array([np.log(p1[i, wifi[iv][i]] + 1e-9) - np.log(p0[i, wifi[iv][i]] + 1e-9)
                     for i in range(len(iv))])

print(f"Wi-Fi 预测器单模型（用真实标签 vs 不用标签的差距见下）", flush=True)
print(f"logratio 与真实满意度的 AUC = {roc_auc_score(y[iv], logratio):.6f}", flush=True)

# 10% rank 加进 Busy53（logratio 是对数似然比、可为负，不能 logit；用 rank 去尺度）
rk = lambda a: rankdata(a) / len(a)
oof53 = np.load("pulled_busy53/oof_stacks.npz")["oof53"]
base_oof = oof53[iv]
blend = 0.9 * rk(base_oof) + 0.1 * rk(logratio)
print(f"Busy53(fold0) = {roc_auc_score(y[iv], base_oof):.6f}", flush=True)
print(f"+10% 标签条件 = {roc_auc_score(y[iv], blend):.6f}  Δ {roc_auc_score(y[iv], blend) - roc_auc_score(y[iv], base_oof):+.6f}", flush=True)

np.savez("cache/label_cond_fold0.npz", val_id=train["id"].iloc[iv].to_numpy(), logratio=logratio, y_val=y[iv])
