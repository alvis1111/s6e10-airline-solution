"""fusion_marginal — B(蒸馏 student)加入自有融合(route+context+tabm)的边际贡献。

三腿在 full ia 训练 -> 预测 iv -> rank 融合(0.3/0.3/0.4)。B 用蒸馏软标签训练。
比较: 融合 vs 融合+B(固定 rank 平均 + 逻辑回归)。
"""
import contextlib, io
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score
from pytabkit import RealMLP_TD_Classifier, TabM_D_Classifier
from pytabkit.models.nn_models.nn import ToSoftLabelLayer
from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface

DATA = "data/playground-series-s6e10"; TGT = "satisfaction"; SEED = 42
REAL = r"C:\Users\GEM07\.cache\kagglehub\datasets\teejmahal20\airline-passenger-satisfaction\versions\1"
train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv")
y = train[TGT].to_numpy(dtype=np.int64); ys = pd.Series(y)
RATING = ["Inflight wifi service","Departure/Arrival time convenient","Ease of Online booking","Gate location",
    "Food and drink","Online boarding","Seat comfort","Inflight entertainment","On-board service","Leg room service",
    "Baggage handling","Checkin service","Cleanliness"]
CATS = ["Gender","Customer Type","Type of Travel","Class"]
ORIG = [c for c in train.columns if c not in ["id","satisfaction"]]

# 原始数据 + orig model
orig = pd.concat([pd.read_csv(f"{REAL}/train.csv"), pd.read_csv(f"{REAL}/test.csv")], ignore_index=True)
orig["satisfaction"] = (orig["satisfaction"]=="satisfied").astype(int)
orig = orig.drop(columns=["Unnamed: 0","id","Inflight service"], errors="ignore")
def rh(df): return pd.util.hash_pandas_object(df[ORIG].astype(str), index=False)
ov = rh(orig).isin(pd.concat([rh(train), rh(test)], ignore_index=True))
orig = orig[~ov].drop_duplicates().reset_index(drop=True)
oy = orig["satisfaction"].astype(int); oX = orig[ORIG].copy()
_v = {}
for c in CATS: _v[c] = sorted(oX[c].dropna().unique()); oX[c] = pd.Categorical(oX[c], categories=_v[c])
for c in ORIG:
    if c not in CATS: oX[c] = oX[c].astype("float32")
om = xgb.XGBClassifier(objective="binary:logistic", n_estimators=600, learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.5, tree_method="hist", enable_categorical=True,
    n_jobs=-1, random_state=0, verbosity=0).fit(oX, oy)
def apply_om(df):
    Xr = df[ORIG].copy()
    for c in CATS: Xr[c] = pd.Categorical(Xr[c], categories=_v[c])
    for c in ORIG:
        if c not in CATS: Xr[c] = Xr[c].astype("float32")
    p = om.predict_proba(Xr)[:,1].astype("float32")
    return p, np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).astype("float32")
train["orig_proba"], train["orig_logit"] = apply_om(train)
_ce = {"Gender":{"Male":1,"Female":0},"Customer Type":{"Loyal Customer":1,"disloyal Customer":0},
    "Type of Travel":{"Business travel":1,"Personal Travel":0},"Class":{"Business":2,"Eco Plus":1,"Eco":0}}
_enc = orig[ORIG].copy()
for c,m in _ce.items(): _enc[c] = orig[c].map(m)
omm = {}
for cols in [("Type of Travel",),("Class",),("Customer Type",),("Type of Travel","Class","Customer Type")]:
    k = _enc[list(cols)].astype(str).apply("|".join, axis=1)
    omm[cols] = pd.DataFrame({"k":k,"l":oy}).groupby("k")["l"].mean()
oprior = float(oy.mean())

def base_preprocess(df):
    df = df.copy(); df.drop(columns=["id"], inplace=True, errors="ignore")
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])
    for c,m in _ce.items(): df[c] = df[c].map(m)
    df["total_delay"] = df["Departure Delay in Minutes"]+df["Arrival Delay in Minutes"]
    df["is_delayed"] = (df["Departure Delay in Minutes"]>0).astype(int)
    df["avg_rating"]=df[RATING].mean(axis=1); df["min_rating"]=df[RATING].min(axis=1)
    df["max_rating"]=df[RATING].max(axis=1); df["rating_std"]=df[RATING].std(axis=1)
    df["zero_count"]=(df[RATING]==0).sum(axis=1); df["has_zero"]=(df["zero_count"]>0).astype(int)
    df["age_group"]=pd.cut(df["Age"],bins=[0,18,30,45,60,100],labels=[0,1,2,3,4]).astype(float)
    df["biz_biz"]=((df["Type of Travel"]==1)&(df["Class"]==2)).astype(int)
    df["distance_bin"]=pd.cut(df["Flight Distance"],bins=[0,500,1500,float("inf")],labels=[0,1,2]).astype(float)
    return df

train_p = base_preprocess(train)
FEATS = [c for c in train_p.columns if c!="satisfaction"]
X_base = train_p[FEATS].reset_index(drop=True)
y = train_p["satisfaction"].reset_index(drop=True)
KEY_PAIRS = [("Class","Type of Travel"),("Customer Type","Type of Travel"),("Class","Customer Type"),
    ("Gender","Class"),("Inflight wifi service","Online boarding"),("Inflight wifi service","Type of Travel"),
    ("Online boarding","Type of Travel"),("Seat comfort","Inflight entertainment"),
    ("Class","Inflight wifi service"),("Class","Online boarding")]
BASE_ENC = ["Gender","Customer Type","Type of Travel","Class","Inflight wifi service",
    "Ease of Online booking","Online boarding","Seat comfort","Inflight entertainment","On-board service","Age","Flight Distance"]
def ck(df, cols):
    v = df[list(cols)].astype(str); k = v.iloc[:,0]
    for c in v.columns[1:]: k = k.str.cat(v[c], sep="|")
    return k
def add_dp(Xt,Xv):
    dc="Flight Distance"; meas=RATING+["Age","Departure Delay in Minutes","Arrival Delay in Minutes"]
    cp=["Gender","Customer Type","Type of Travel","Class"]
    g=Xt.groupby(dc)[meas].agg(["mean","std"]); cnt=Xt[dc].value_counts()
    prop={c:pd.crosstab(Xt[dc],Xt[c],normalize="index") for c in cp}
    dd={c:{"mean":Xt[c].mean(),"std":Xt[c].std()} for c in meas}; out=[]
    for fr in [Xt,Xv]:
        f={"profile_route_count":fr[dc].map(cnt).fillna(0).astype("float32")}
        for c in meas:
            fn=c.lower().replace(" ","_").replace("/","_").replace("-","_")
            mu=fr[dc].map(g[(c,"mean")]).fillna(dd[c]["mean"]).astype("float32")
            sg=fr[dc].map(g[(c,"std")]).fillna(dd[c]["std"]).astype("float32")
            f[f"profile_{fn}_mean"]=mu; f[f"profile_{fn}_std"]=sg; f[f"profile_{fn}_residual"]=(fr[c]-mu).astype("float32")
        for c,tb in prop.items():
            pr=Xt[c].value_counts(normalize=True); fn=c.lower().replace(" ","_").replace("/","_").replace("-","_")
            for cat in tb.columns:
                f[f"profile_{fn}_{cat}_share"]=fr[dc].map(tb[cat]).fillna(float(pr.get(cat,0))).astype("float32")
        out.append(pd.concat([fr, pd.DataFrame(f, index=fr.index)], axis=1))
    return out
def add_freq(Xt,Xv):
    specs=[(c,) for c in BASE_ENC]+KEY_PAIRS; out=[Xt.copy(),Xv.copy()]
    for cols in specs:
        nm="__".join(c.lower().replace(" ","_").replace("/","_") for c in cols)
        fm=ck(Xt,cols).value_counts(normalize=True)
        for i,fr in enumerate([Xt,Xv]):
            fq=ck(fr,cols).map(fm).fillna(0).astype("float32")
            out[i][f"freq_{nm}"]=fq; out[i][f"rarity_{nm}"]=-np.log(fq.clip(lower=1.0/len(Xt))).astype("float32")
    return out
def add_om(Xt,Xv):
    out=[Xt.copy(),Xv.copy()]
    for cols,mp in omm.items():
        ft="orig_mean_"+"__".join(c.lower().replace(" ","_") for c in cols)
        for i,fr in enumerate([Xt,Xv]):
            k=fr[list(cols)].astype(str).apply("|".join,axis=1)
            out[i][ft]=k.map(mp).fillna(oprior).astype("float32")
    return out
def add_te(Xt,Xv,yt,fold):
    def bk(df):
        ks={c:df[c] for c in BASE_ENC}
        for cols in KEY_PAIRS:
            nm="pair_"+"__".join(c.lower().replace(" ","_") for c in cols)
            ks[nm]=df[cols[0]]*10+df[cols[1]] if all(c in RATING for c in cols) else ck(df,cols).astype("category")
        ks["distance_250_bin"]=(df["Flight Distance"]//250).astype("float32"); ks["age_5_bin"]=(df["Age"]//5).astype("float32")
        return pd.DataFrame(ks,index=df.index)
    enc=TargetEncoder(target_type="binary",smooth=20.0,cv=5,shuffle=True,random_state=1042+fold)
    tr=enc.fit_transform(bk(Xt),yt); va=enc.transform(bk(Xv))
    nm=[f"te_{c}" for c in bk(Xt).columns]
    return (pd.concat([Xt,pd.DataFrame(tr.astype("float32"),columns=nm,index=Xt.index)],axis=1),
            pd.concat([Xv,pd.DataFrame(va.astype("float32"),columns=nm,index=Xv.index)],axis=1))

def route_feats(Xt, Xv, yt, fold):
    Xt, Xv = add_dp(Xt, Xv); Xt, Xv = add_freq(Xt, Xv); Xt, Xv = add_om(Xt, Xv); Xt, Xv = add_te(Xt, Xv, yt, fold)
    for f in [Xt, Xv]:
        f.replace([np.inf,-np.inf],0,inplace=True); f.fillna(0,inplace=True)
        f = f.loc[:,~f.columns.duplicated()].astype("float32")
    Xt = Xt[[c for c in Xt.columns if Xt[c].nunique()>1]]; Xv = Xv[Xt.columns]
    return Xt, Xv

# 外层 fold 1
outer = list(StratifiedKFold(10, shuffle=True, random_state=42).split(X_base, y))
ia, iv = outer[1]
print(f"fold 1: ia={len(ia)} iv={len(iv)}", flush=True)
y_ia, y_iv = y.iloc[ia], y.iloc[iv]

# ---- 三腿 full ia -> iv ----
lgb_p = dict(objective="binary", metric="auc", n_estimators=1500, learning_rate=0.04, num_leaves=63, max_depth=6,
    subsample=0.9, subsample_freq=1, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=5.0, min_child_samples=20, n_jobs=-1, verbosity=-1)
xgb_p = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", n_estimators=1500, learning_rate=0.04,
    max_depth=6, min_child_weight=5, subsample=0.9, colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=5.0, max_bin=256,
    n_jobs=-1, verbosity=0)

# route
Xt, Xv = route_feats(X_base.iloc[ia], X_base.iloc[iv], y_ia, 1)
route_iv = np.zeros(len(iv))
for s in [42, 7, 2024]:
    ml = lgb.LGBMClassifier(**{**lgb_p, "random_state": s}); ml.fit(Xt, y_ia)
    route_iv += ml.predict_proba(Xv)[:,1]/3
    mx = xgb.XGBClassifier(**{**xgb_p, "random_state": s}); mx.fit(Xt, y_ia)
    route_iv += mx.predict_proba(Xv)[:,1]/3
print(f"route iv AUC={roc_auc_score(y_iv, route_iv):.5f}", flush=True)

# context (条件TE+频率 LGBM)
def cond_feats(df):
    seg = df['Type of Travel'].astype(str)+'|'+df['Customer Type'].astype(str)+'|'+df['Class'].astype(str)
    grp = df['Type of Travel'].astype(str)+'|'+df['Customer Type'].astype(str)
    z = pd.DataFrame(index=df.index)
    for c in RATING: z[c+'_segment'] = df[c].astype(str)+'|'+seg
    for c,d in [('Inflight wifi service','Online boarding'),('Inflight wifi service','Ease of Online booking'),
                ('Online boarding','Seat comfort'),('Seat comfort','Inflight entertainment'),('Inflight entertainment','Cleanliness')]:
        z[c+'__'+d] = df[c].astype(str)+'|'+df[d].astype(str)+'|'+grp
    return z
def num_grp(df):
    return pd.DataFrame({f'{c}__{g}': df[c].astype(str)+'|'+df[g].astype(str)
        for c in ['Age','Flight Distance'] for g in ['Type of Travel','Customer Type','Class']}, index=df.index)
ENC_AUTO = dict(target_type='binary', smooth='auto', cv=5, shuffle=True, random_state=20261002)
raw = train.drop(columns=['id','satisfaction']).astype(str)
Z_cond = cond_feats(train); Z_num = train[['Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes']].astype(str)
G = num_grp(train)
Xc = train.drop(columns=['id','satisfaction']).copy(); Xc['traveler_segment'] = Xc['Type of Travel']+'|'+Xc['Customer Type']+'|'+Xc['Class']
Xc['arrival_missing'] = Xc['Arrival Delay in Minutes'].isna().astype('float32')
for c in CATS+RATING+['traveler_segment']: Xc[c] = Xc[c].astype(str).astype(object)
LES = {c: LabelEncoder().fit(Xc[c]) for c in CATS+RATING+['traveler_segment']}
Xct, Xcv = Xc.iloc[ia].copy(), Xc.iloc[iv].copy()
for f in [Xct, Xcv]:
    for c in ['Age','Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes']:
        f[c] = f[c].fillna(Xct[c].median()).astype('float32')
    f['arrival_missing'] = f['arrival_missing'].astype('float32')
for c in CATS+RATING+['traveler_segment']:
    Xct[c] = LES[c].transform(Xct[c]); Xcv[c] = LES[c].transform(Xcv[c])
def te_tr(enc, Z, tr, va):
    return enc.fit_transform(Z.iloc[tr], y.iloc[tr]).astype('float32'), enc.transform(Z.iloc[va]).astype('float32')
for src, nm in [(raw,'te'),(Z_cond,'cond_te')]:
    enc = TargetEncoder(**ENC_AUTO); et, ev = te_tr(enc, src, ia, iv)
    for i in range(et.shape[1]): Xct[f'{nm}_{i}']=et[:,i]; Xcv[f'{nm}_{i}']=ev[:,i]
enc = TargetEncoder(**dict(ENC_AUTO, smooth=20)); et, ev = te_tr(enc, G, ia, iv)
for i in range(et.shape[1]): Xct[f'ng_te_{i}']=et[:,i]; Xcv[f'ng_te_{i}']=ev[:,i]
Zf = pd.concat([Z_num, G], axis=1)
def crossfit_freq(Z, tr, va):
    z = Z.iloc[tr].reset_index(drop=True)
    def transform(fit, apply):
        v = np.empty((len(apply), fit.shape[1]), dtype='float32')
        for j,col in enumerate(fit):
            d = fit[col].value_counts(dropna=False)/len(fit); v[:,j] = np.log1p(apply[col].map(d).fillna(0).to_numpy()*1e6)
        return v
    return transform(z, Z.iloc[tr]), transform(z, Z.iloc[va])
fv, ff = crossfit_freq(Zf, ia, iv)
for i in range(fv.shape[1]): Xct[f'freq_{i}']=fv[:,i]; Xcv[f'freq_{i}']=ff[:,i]
mc = lgb.LGBMClassifier(objective="binary", metric="auc", learning_rate=0.03, num_leaves=63, min_child_samples=50,
    colsample_bytree=0.7, subsample=0.8, subsample_freq=1, reg_alpha=0.3, reg_lambda=1.0, n_estimators=1500,
    random_state=42, n_jobs=-1, verbosity=-1)
mc.fit(Xct, y_ia); context_iv = mc.predict_proba(Xcv)[:,1]
print(f"context iv AUC={roc_auc_score(y_iv, context_iv):.5f}", flush=True)

# tabm
Xt, Xv = route_feats(X_base.iloc[ia], X_base.iloc[iv], y_ia, 1)
MP = dict(device='cuda:0', random_state=3407, n_cv=1, n_refit=0, n_threads=6, verbosity=0,
    val_metric_name='1-auc_ovr', n_epochs=128, batch_size=1024, tabm_k=16, num_emb_type='pwl',
    num_emb_n_bins=32, d_embedding=16, d_block=256, n_blocks=2, dropout=.1, patience=16,
    allow_amp=True, compile_model=False, share_training_batches=False)
mt = TabM_D_Classifier(**MP); mt.fit(Xt, y_ia); tabm_iv = mt.predict_proba(Xv)[:,1]
print(f"tabm iv AUC={roc_auc_score(y_iv, tabm_iv):.5f}", flush=True)
del mt; torch.cuda.empty_cache()

# rank 融合
rk = lambda p: rankdata(p)/len(p)
fusion_iv = 0.3*rk(route_iv) + 0.3*rk(context_iv) + 0.4*rk(tabm_iv)
print(f"\n自有融合 iv AUC = {roc_auc_score(y_iv, fusion_iv):.6f}", flush=True)

# B (蒸馏 student) iv 预测
teacher_prob = None  # 用 fold1 教师 OOF 重建软标签
rte = np.load("cache/teacher_route_oof.npy")[ia]; ctx = np.load("cache/teacher_context_oof.npy")[ia]; tab = np.load("cache/teacher_tabm_oof.npy")[ia]
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip'); tp = iso.fit_transform(0.3*rk(rte)+0.3*rk(ctx)+0.4*rk(tab), y_ia.to_numpy())

# lean RealMLP 输入
Xlean = train.drop(columns=['id','satisfaction']).copy()
lcat = Xlean.select_dtypes(exclude="number").columns.tolist()
lnum = Xlean.select_dtypes(include="number").columns.tolist()
for c in lcat: Xlean[c] = Xlean[c].fillna("missing")
for c in lnum: Xlean[c] = Xlean[c].fillna(0.0)
for c in lnum:
    cn = f"{c}_cat_"
    codes, uniques = Xlean[c].factorize()
    Xlean[cn] = codes; Xlean[cn] = Xlean[cn].astype("category")
for c in lcat + ["Leg room service"]:
    cn = f"_{c}_count"
    cm = Xlean[c].value_counts()
    Xlean[cn] = Xlean[c].map(cm).fillna(0).astype("int32")
for cols in [("Class","Type of Travel","Gender"),("Class","Type of Travel","Age"),("Flight Distance","Departure/Arrival time convenient")]:
    cn = "_".join(cols) + "_"
    s = Xlean[cols[0]].astype(str)
    for c in cols[1:]: s = s + "_" + Xlean[c].astype(str)
    codes, uniques = pd.factorize(s, sort=False)
    Xlean[cn] = codes; Xlean[cn] = Xlean[cn].astype("category")

# 折内 TE + drop(对齐 realmlp_repro_lean.py 最后两步)
combos = ["Class_Type of Travel_Gender_", "Class_Type of Travel_Age_", "Flight Distance_Departure/Arrival time convenient_"]
enc_lean = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=SEED)
tn = [f"_{c}TE" for c in combos]
Xlean_te = np.zeros((len(Xlean), len(tn)), dtype=np.float32)
Xlean_te[ia] = enc_lean.fit_transform(Xlean[combos].iloc[ia], ys.iloc[ia])
Xlean_te[iv] = enc_lean.transform(Xlean[combos].iloc[iv])
Xlean = pd.concat([Xlean, pd.DataFrame(Xlean_te, columns=tn, index=Xlean.index)], axis=1)
Xlean = Xlean.drop(["Class_Type of Travel_Age_", "Flight Distance_Departure/Arrival time convenient_"], axis=1)
params = dict(random_state=SEED, verbosity=0, val_metric_name="1-auc_ovr", n_ens=8, n_epochs=3, batch_size=256,
    use_early_stopping=False, lr=0.053, wd=0.015, sq_mom=0.988, lr_sched="flat_anneal", wd_sched="cos_log_15",
    first_layer_lr_factor=0.25, embedding_size=5, max_one_hot_cat_size=18, hidden_sizes=[512,256,128], act="silu",
    p_drop=0.05, p_drop_sched="invsqrtp1e-3", plr_hidden_1=16, plr_hidden_2=8, plr_act_name="gelu", plr_lr_factor=0.1151,
    plr_sigma=2.33, ls_eps=0.0, ls_eps_sched="sqrt_cos", add_front_scale=False, bias_init_mode="neg-uniform-dynamic-2",
    tfms=["one_hot","median_center","robust_scale","smooth_clip","embedding","l2_normalize"])
params["stop_epoch"] = params["n_epochs"]

_orig_forward = ToSoftLabelLayer.forward_tensors
def patched_forward(self, tensors):
    if 'y' in tensors and torch.is_floating_point(tensors['y']): return tensors
    return _orig_forward(self, tensors)
ToSoftLabelLayer.forward_tensors = patched_forward
_SOFT = None
class SoftNN(NNAlgInterface):
    def fit(self, ds=None, *args, **kwargs):
        if _SOFT is not None and ds is not None: ds.tensors['y'] = torch.as_tensor(_SOFT, dtype=torch.float32)
        return super().fit(ds=ds, *args, **kwargs)
class SoftRealMLP(RealMLP_TD_Classifier):
    def _create_alg_interface(self, n_cv): return SoftNN(**self.get_config())

def run_B(soft):
    global _SOFT; _SOFT = soft
    m = SoftRealMLP(**params)
    with contextlib.redirect_stdout(io.StringIO()): m.fit(Xlean.iloc[ia], ys.iloc[ia], Xlean.iloc[iv], ys.iloc[iv])
    return m.predict_proba(Xlean.iloc[iv])[:,1]

s_soft = np.zeros((len(ia),2), dtype=np.float32); s_soft[:,1] = 0.8*y_ia.to_numpy() + 0.2*tp; s_soft[:,0] = 1-s_soft[:,1]
B_iv = run_B(s_soft)
print(f"B iv AUC={roc_auc_score(y_iv, B_iv):.6f}", flush=True)

# 融合 + B 的边际
lr_stack = np.full(len(iv), np.nan)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold as SKF
S = np.stack([fusion_iv, B_iv], axis=1)
for tr, va in SKF(5, shuffle=True, random_state=42).split(S, y_iv):
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(S[tr], y_iv.iloc[tr]); lr_stack[va] = lr.predict_proba(S[va])[:,1]
rank_avg = rk(fusion_iv) + rk(B_iv)

print(f"\n=== 边际贡献 ===")
print(f"融合 iv AUC        = {roc_auc_score(y_iv, fusion_iv):.6f}")
print(f"融合+B (rank平均)  = {roc_auc_score(y_iv, rank_avg):.6f}  Δ={roc_auc_score(y_iv, rank_avg)-roc_auc_score(y_iv, fusion_iv):+.6f}")
print(f"融合+B (LR stack)  = {roc_auc_score(y_iv, lr_stack):.6f}  Δ={roc_auc_score(y_iv, lr_stack)-roc_auc_score(y_iv, fusion_iv):+.6f}")
