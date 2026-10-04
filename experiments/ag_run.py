"""S6E10 — AutoGluon AutoML: 自动特征 + 多模型 + stacking."""
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

DATA = "data/playground-series-s6e10"
TARGET = "satisfaction"

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
SERVICE_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]

# 传入我们已知的好特征, AutoGluon 在其上继续做自己的 FE 和 stacking
def add_feats(df):
    df = df.copy()
    df["service_sum"] = df[SERVICE_COLS].sum(axis=1)
    df["service_mean"] = df[SERVICE_COLS].mean(axis=1)
    df["service_std"] = df[SERVICE_COLS].std(axis=1)
    df["total_delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    df["is_delayed"] = (df["total_delay"] > 0).astype(int)
    df["travel_x_class"] = df["Type of Travel"].astype(str) + "_" + df["Class"].astype(str)
    df["travel_x_customer"] = df["Type of Travel"].astype(str) + "_" + df["Customer Type"].astype(str)
    df["dist_bucket"] = pd.cut(
        df["Flight Distance"], bins=[0, 500, 1000, 2000, 3000, 5000, np.inf], labels=False)
    return df

train = add_feats(train)
test = add_feats(test)

for c in CAT_COLS + ["travel_x_class", "travel_x_customer"]:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype("category")
train["dist_bucket"] = train["dist_bucket"].astype("category")
test["dist_bucket"] = test["dist_bucket"].astype("category")

# 目标转 int, 去掉 id
train[TARGET] = train[TARGET].astype(int)
train = train.drop(columns=["id"])
test_ids = test["id"]
test = test.drop(columns=["id"])

print(f"train {train.shape}, test {test.shape}", flush=True)

predictor = TabularPredictor(label=TARGET, eval_metric="roc_auc", path="ag_models").fit(
    train, time_limit=3600, presets="best_quality",
)

# 预测正类概率
pred_proba = predictor.predict_proba(test)
pos_col = pred_proba.columns[-1]  # 1 或 True
sub = pd.DataFrame({"id": test_ids, TARGET: pred_proba[pos_col].astype(float)})
sub.to_csv("submission_autogluon.csv", index=False)
print(f"submission saved -> submission_autogluon.csv  (pos_col={pos_col})", flush=True)

print("\n=== leaderboard ===", flush=True)
print(predictor.leaderboard(silent=True).to_string(), flush=True)
