"""原始数据逐列 P(satisfied|value) lookup：补齐缺失的 18 个单列条件率（只算特征+覆盖报告，不训练）。

= 当前主模型 route_profile_aux.py 的 orig_mean_maps（只覆盖 Type of Travel/Class/Customer Type 3 单列
  + 1 三元）扩展到全部 21 特征的单列。清洗与主模型一致：去重叠 + 去重；只用外部原始标签算条件率；
  未出现值回退 orig_prior；首轮均值映射、不搜平滑。NaN 作为独立键（Arrival Delay 的 on-time 语义自然对齐）。

输出：cache/orig_lookup18_train.npy / orig_lookup18_test.npy（各 18 列）+ 覆盖度报告。
"""
import numpy as np
import pandas as pd

DATA = "data/playground-series-s6e10"
REAL = r"C:\Users\GEM07\.cache\kagglehub\datasets\teejmahal20\airline-passenger-satisfaction\versions\1"

RATING_COLS = ["Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
    "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness"]
CAT_COLS_RAW = ["Gender", "Customer Type", "Type of Travel", "Class"]
ORIGINAL_FEATURES = (["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
                     + RATING_COLS + CAT_COLS_RAW)

# 缺失的 18 个单列 = 全部 21 特征 - 已有 3 个单列(Type of Travel/Class/Customer Type)
MISSING_18 = (["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
              + RATING_COLS + ["Gender"])
assert len(MISSING_18) == 18 and len(ORIGINAL_FEATURES) == 21

_cat_encode = {"Gender": {"Male": 1, "Female": 0}}

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")

# ---- 原始数据清洗（与 route_profile_aux 一致：去重叠 + 去重） ----
orig_tr = pd.read_csv(f"{REAL}/train.csv")
orig_te = pd.read_csv(f"{REAL}/test.csv")
original = pd.concat([orig_tr, orig_te], ignore_index=True)
original["satisfaction"] = (original["satisfaction"] == "satisfied").astype(int)
original = original.drop(columns=["Unnamed: 0", "id", "Inflight service"], errors="ignore")


def row_hashes(df):
    return pd.util.hash_pandas_object(df[ORIGINAL_FEATURES].astype(str), index=False)


comp_hashes = pd.concat([row_hashes(train), row_hashes(test)], ignore_index=True)
orig_hashes = row_hashes(original)
overlap = orig_hashes.isin(comp_hashes)
orig_clean = original[~overlap].drop_duplicates().reset_index(drop=True)
orig_y = orig_clean["satisfaction"].astype(int)
orig_prior = float(orig_y.mean())
print(f"原始数据: {len(original)} -> 去重叠去重后 {len(orig_clean)}  prior={orig_prior:.4f}", flush=True)


def key_of(df, col):
    """构造与主模型 add_orig_means 一致的键：类别列映射 0/1，其余取原值；NaN 保留为 'nan'。"""
    if col in _cat_encode:
        return df[col].map(_cat_encode[col]).astype(str)
    return df[col].astype(str)


# ---- 计算 18 个单列 lookup（原始数据上的均值映射） ----
lookup = {}
for col in MISSING_18:
    k = key_of(orig_clean, col)
    lookup[col] = pd.DataFrame({"key": k, "label": orig_y}).groupby("key")["label"].mean()

# ---- 应用到竞赛 train/test（原值，未做 base_preprocess 填充） ----
def apply_lookup(df, out):
    for j, col in enumerate(MISSING_18):
        k = key_of(df, col)
        out[:, j] = k.map(lookup[col]).fillna(orig_prior).astype("float32")


lut_tr = np.zeros((len(train), 18), dtype="float32")
lut_te = np.zeros((len(test), 18), dtype="float32")
apply_lookup(train, lut_tr)
apply_lookup(test, lut_te)

np.save("cache/orig_lookup18_train.npy", lut_tr)
np.save("cache/orig_lookup18_test.npy", lut_te)

# ---- 覆盖度报告 ----
print(f"\n{'列':40s} {'原始基数':>8s} {'训练回退率':>10s} {'测试回退率':>10s}", flush=True)
for j, col in enumerate(MISSING_18):
    card = len(lookup[col])
    tr_fb = (lut_tr[:, j] == orig_prior).mean()          # 近似：等于 prior 即回退（可能掩盖恰好等于 prior 的值）
    te_fb = (lut_te[:, j] == orig_prior).mean()
    print(f"{col:40s} {card:8d} {tr_fb:10.4f} {te_fb:10.4f}", flush=True)

# 抽查：Age 与 Gender 的前几档
print("\n抽查 Age lookup 前 8 档:", flush=True)
print(lookup["Age"].head(8).to_string(), flush=True)
print("抽查 Gender lookup:", flush=True)
print(lookup["Gender"].to_string(), flush=True)
print("\nsaved -> cache/orig_lookup18_train.npy / orig_lookup18_test.npy", flush=True)
