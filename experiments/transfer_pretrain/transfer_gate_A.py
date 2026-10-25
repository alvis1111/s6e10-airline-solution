"""transfer_gate_A — 冻结特征工程(在 ia 拟合)+ A 臂复现验证。

特征映射(factorize/count/combo/TE/缺失)全部在比赛外层训练折 ia 上拟合后冻结,
供 A/B/C 和原始数据共用。本脚本只跑 A(lean RealMLP 从头训), 验证 ~0.9616。
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
import contextlib, io
from pytabkit import RealMLP_TD_Classifier

DATA = "data/playground-series-s6e10"; TGT = "satisfaction"; SEED = 42
REAL = r"C:\Users\GEM07\.cache\kagglehub\datasets\teejmahal20\airline-passenger-satisfaction\versions\1"

train = pd.read_csv(f"{DATA}/train.csv")
orig = pd.concat([pd.read_csv(f"{REAL}/train.csv"), pd.read_csv(f"{REAL}/test.csv")], ignore_index=True)
orig["satisfaction"] = (orig["satisfaction"] == "satisfied").astype(int)
orig = orig.drop(columns=["Unnamed: 0", "id", "Inflight service"], errors="ignore")

# 外层 fold 0
y = train[TGT].to_numpy(dtype=np.int64)
outer = list(StratifiedKFold(10, shuffle=True, random_state=42).split(train.drop(columns=[TGT]), y))
ia, iv = outer[0]

CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
IMPORTANT_COMBOS = [("Class","Type of Travel","Gender"), ("Class","Type of Travel","Age"),
                    ("Flight Distance","Departure/Arrival time convenient")]

def fe_fit_apply(df, ia, fit=True):
    """lean 配方特征工程: 在 ia 上拟合映射, 应用到整个 df(冻结)。"""
    df = df.drop(columns=["id", "satisfaction"], errors="ignore").copy()
    cat_cols = [c for c in CATS if c in df.columns]
    num_cols = [c for c in df.columns if c not in cat_cols + ["satisfaction"]]
    if fit:
        # 在 ia 上拟合
        ia_df = df.iloc[ia]
        # 缺失处理
        for c in cat_cols: df[c] = df[c].fillna("missing")
        for c in num_cols: df[c] = df[c].fillna(0.0)
        # 数值类别副本(factorize 在 ia)
        maps = {}
        for c in num_cols:
            cn = f"{c}_cat_"
            codes, uniques = pd.factorize(ia_df[c].dropna())
            maps[c] = uniques
            df[cn] = df[c].map({v: i for i, v in enumerate(uniques)}).fillna(-1).astype("int32")
            df[cn] = df[cn].astype("category")
        # count 编码(value_counts 在 ia)
        for c in cat_cols + ["Leg room service"]:
            cn = f"_{c}_count"
            cm = ia_df[c].value_counts()
            maps[cn] = cm
            df[cn] = df[c].map(cm).fillna(0).astype("int32")
        # combo 编码(factorize 在 ia)
        combo_names = []
        for cols in IMPORTANT_COMBOS:
            cn = "_".join(cols) + "_"; combo_names.append(cn)
            s_ia = ia_df[cols[0]].astype(str)
            for c in cols[1:]: s_ia = s_ia + "_" + ia_df[c].astype(str)
            codes, uniques = pd.factorize(s_ia, sort=False)
            maps[cn] = uniques
            s = df[cols[0]].astype(str)
            for c in cols[1:]: s = s + "_" + df[c].astype(str)
            df[cn] = s.map({v: i for i, v in enumerate(uniques)}).fillna(-1).astype("int32")
            df[cn] = df[cn].astype("category")
        return df, combo_names, maps, cat_cols, num_cols
    return df

# 拟合 + 应用(在 ia 拟合, 应用到全量)
Xfe, combo_names, maps, cat_cols, num_cols = fe_fit_apply(train, ia, fit=True)

# TE(在 ia 拟合, 内层 CV for ia 行; iv 行用 ia 全量 fit 的编码器)
te_names = [f"_{c}TE" for c in combo_names]
Xa, Xv = Xfe.iloc[ia].copy(), Xfe.iloc[iv].copy()
enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=SEED)
Xa[te_names] = enc.fit_transform(Xa[combo_names], y[ia])
Xv[te_names] = enc.transform(Xv[combo_names])
drop = ["Class_Type of Travel_Age_", "Flight Distance_Departure/Arrival time convenient_"]
Xa, Xv = Xa.drop(drop, axis=1), Xv.drop(drop, axis=1)

params = dict(random_state=SEED, verbosity=0, val_metric_name="1-auc_ovr", n_ens=8, n_epochs=3, batch_size=256,
    use_early_stopping=False, lr=0.053, wd=0.015, sq_mom=0.988, lr_sched="flat_anneal", wd_sched="cos_log_15",
    first_layer_lr_factor=0.25, embedding_size=5, max_one_hot_cat_size=18, hidden_sizes=[512,256,128], act="silu",
    p_drop=0.05, p_drop_sched="invsqrtp1e-3", plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151,
    plr_sigma=2.33, ls_eps=0.0, ls_eps_sched="sqrt_cos", add_front_scale=False, bias_init_mode="neg-uniform-dynamic-2",
    tfms=["one_hot","median_center","robust_scale","smooth_clip","embedding","l2_normalize"])
params["stop_epoch"] = params["n_epochs"]

m = RealMLP_TD_Classifier(**params)
with contextlib.redirect_stdout(io.StringIO()):
    m.fit(Xa, y[ia], Xv, y[iv])
auc_A = roc_auc_score(y[iv], m.predict_proba(Xv)[:, 1])
print(f"\n[A] 冻结特征工程 + lean RealMLP: fold0 AUC = {auc_A:.6f}")
print(f"复现目标 ~0.9616, 判定: {'通过 ✅' if abs(auc_A - 0.9616) < 0.0005 else '需检查 ❌'}")
