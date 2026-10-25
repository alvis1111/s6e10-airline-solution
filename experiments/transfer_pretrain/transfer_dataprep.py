"""transfer_dataprep — 冻结特征工程 + 原始数据转换 + 四项检查。

特征映射(factorize/count/combo/TE/缺失)在 ia 拟合后冻结, 应用到 ia/iv/原始数据。
检查: 重叠排除、未知类别比例、标签/NaN、特征一致性。
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

DATA = "data/playground-series-s6e10"; TGT = "satisfaction"; SEED = 42
REAL = r"C:\Users\GEM07\.cache\kagglehub\datasets\teejmahal20\airline-passenger-satisfaction\versions\1"

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
FEATURES = [c for c in train.columns if c not in ["id", "satisfaction"]]
orig = pd.concat([pd.read_csv(f"{REAL}/train.csv"), pd.read_csv(f"{REAL}/test.csv")], ignore_index=True)
orig["satisfaction"] = (orig["satisfaction"] == "satisfied").astype(int)
orig = orig.drop(columns=["Unnamed: 0", "id", "Inflight service"], errors="ignore")
orig = orig[FEATURES + ["satisfaction"]]  # 重排到比赛特征顺序

y = train[TGT].to_numpy(dtype=np.int64)
outer = list(StratifiedKFold(10, shuffle=True, random_state=42).split(train[FEATURES], y))
ia, iv = outer[0]

CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
COMBOS = [("Class","Type of Travel","Gender"), ("Class","Type of Travel","Age"),
          ("Flight Distance","Departure/Arrival time convenient")]

# ---- 重叠检查(原始 vs 外层验证) ----
def rh(df): return pd.util.hash_pandas_object(df[FEATURES].astype(str), index=False)
iv_hash = set(rh(train.iloc[iv]).tolist())
orig_hash = rh(orig)
overlap_mask = orig_hash.isin(iv_hash)
print(f"[检查4] 原始数据与外层验证精确重合: {int(overlap_mask.sum())} 行(应排除)")
orig = orig[~overlap_mask].reset_index(drop=True)
print(f"        排除后原始数据: {len(orig)} 行")

# ---- 冻结特征工程(在 ia 拟合) ----
def fit_fe_on_ia(train):
    df = train.drop(columns=["id", "satisfaction"]).copy()
    ia_df = df.iloc[ia]
    cat_cols = CATS
    num_cols = [c for c in df.columns if c not in cat_cols]
    for c in cat_cols: df[c] = df[c].fillna("missing")
    for c in num_cols: df[c] = df[c].fillna(0.0)
    maps = {}
    for c in num_cols:
        cn = f"{c}_cat_"
        _, uniques = pd.factorize(ia_df[c].dropna()); maps[c] = uniques
    for c in cat_cols + ["Leg room service"]:
        cn = f"_{c}_count"
        maps[cn] = ia_df[c].value_counts()
    combo_names = []
    for cols in COMBOS:
        cn = "_".join(cols) + "_"; combo_names.append(cn)
        s_ia = ia_df[cols[0]].astype(str)
        for c in cols[1:]: s_ia = s_ia + "_" + ia_df[c].astype(str)
        _, uniques = pd.factorize(s_ia, sort=False); maps[cn] = uniques
    return maps, combo_names, cat_cols, num_cols

maps, combo_names, cat_cols, num_cols = fit_fe_on_ia(train)

def apply_fe(df, maps):
    df = df.drop(columns=["id", "satisfaction"], errors="ignore").copy()
    for c in cat_cols: df[c] = df[c].fillna("missing")
    for c in num_cols: df[c] = df[c].fillna(0.0)
    unknown_ratio = {}
    for c in num_cols:
        cn = f"{c}_cat_"
        uniques = maps[c]
        codes = df[c].map({v: i for i, v in enumerate(uniques)})
        unknown_ratio[cn] = float(codes.isna().mean())
        df[cn] = pd.Categorical(codes.fillna(-1).astype("int32"), categories=np.arange(-1, len(uniques)))
    for c in cat_cols + ["Leg room service"]:
        cn = f"_{c}_count"
        cm = maps[cn]
        df[cn] = df[c].map(cm).fillna(0).astype("int32")
    for cols in COMBOS:
        cn = "_".join(cols) + "_"
        uniques = maps[cn]
        s = df[cols[0]].astype(str)
        for c in cols[1:]: s = s + "_" + df[c].astype(str)
        codes = s.map({v: i for i, v in enumerate(uniques)})
        unknown_ratio[cn] = float(codes.isna().mean())
        df[cn] = pd.Categorical(codes.fillna(-1).astype("int32"), categories=np.arange(-1, len(uniques)))
    return df, unknown_ratio

train_fe, _ = apply_fe(train, maps)
orig_fe, orig_unk = apply_fe(orig, maps)

print(f"\n[检查2] 原始数据未知类别比例(前若干列, 重点距离/交叉):")
for k, v in sorted(orig_unk.items(), key=lambda x: -x[1])[:10]:
    print(f"  {k}: {v:.4f}")

# ---- TE(ia 拟合) ----
te_names = [f"_{c}TE" for c in combo_names]
Xa, Xv = train_fe.iloc[ia].copy(), train_fe.iloc[iv].copy()
enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=SEED)
Xa[te_names] = enc.fit_transform(Xa[combo_names], y[ia])
Xv[te_names] = enc.transform(Xv[combo_names])
O = orig_fe.copy()
O[te_names] = enc.transform(O[combo_names])
drop = ["Class_Type of Travel_Age_", "Flight Distance_Departure/Arrival time convenient_"]
Xa, Xv, O = Xa.drop(drop, axis=1), Xv.drop(drop, axis=1), O.drop(drop, axis=1)

# ---- 检查1: 特征名/顺序/类型一致 ----
print(f"\n[检查1] 特征一致性: train={Xa.shape[1]} 列, valid={Xv.shape[1]} 列, original={O.shape[1]} 列")
print(f"        列名完全一致: {list(Xa.columns)==list(Xv.columns)==list(O.columns)}")

# ---- 检查3: 标签/NaN/Inf ----
print(f"\n[检查3] 标签含义: 比赛 train 正例率={y[ia].mean():.4f}, 原始数据正例率={orig['satisfaction'].mean():.4f}")
print(f"        比赛转换后 NaN 列数: {int(Xa.isna().any().sum())}, Inf 数: {int(np.isinf(Xa.select_dtypes(include=np.number).to_numpy()).sum())}")
print(f"        原始数据转换后 NaN 列数: {int(O.isna().any().sum())}, Inf 数: {int(np.isinf(O.select_dtypes(include=np.number).to_numpy()).sum())}")

# 保存
np.savez("cache/transfer_dataprep.npz", ia=ia, iv=iv)
Xa.to_pickle("cache/transfer_Xa.pkl"); Xv.to_pickle("cache/transfer_Xv.pkl"); O.to_pickle("cache/transfer_O.pkl")
orig[["satisfaction"]].to_pickle("cache/transfer_oy.pkl")
print(f"\n已保存: Xa={Xa.shape} Xv={Xv.shape} O={O.shape}")
