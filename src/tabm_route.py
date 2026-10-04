"""S6E10 — TabM 第三条路线: 航线画像+原始数据特征, NN 范式."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
import torch
from pytabkit import TabM_D_Classifier

DATA = "data/playground-series-s6e10"
REAL = r"C:\Users\GEM07\.cache\kagglehub\datasets\teejmahal20\airline-passenger-satisfaction\versions\1"

train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv")
y = train['satisfaction'].to_numpy(dtype=np.int64)
RATING_COLS = ["Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink", "Online boarding",
    "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service",
    "Baggage handling", "Checkin service", "Cleanliness"]
CAT_COLS_RAW = ["Gender", "Customer Type", "Type of Travel", "Class"]
ORIGINAL_FEATURES = [c for c in train.columns if c not in ["id", "satisfaction"]]

# 原始数据 + orig 模型
orig = pd.concat([pd.read_csv(f"{REAL}/train.csv"), pd.read_csv(f"{REAL}/test.csv")], ignore_index=True)
orig["satisfaction"] = (orig["satisfaction"]=="satisfied").astype(int)
orig = orig.drop(columns=["Unnamed: 0","id","Inflight service"], errors="ignore")
def rh(df): return pd.util.hash_pandas_object(df[ORIGINAL_FEATURES].astype(str), index=False)
ov = rh(orig).isin(pd.concat([rh(train), rh(test)], ignore_index=True))
orig = orig[~ov].drop_duplicates().reset_index(drop=True)
oy = orig["satisfaction"].astype(int); oX = orig[ORIGINAL_FEATURES].copy()
_v = {}
for c in CAT_COLS_RAW:
    _v[c] = sorted(oX[c].dropna().unique().tolist()); oX[c] = pd.Categorical(oX[c], categories=_v[c])
for c in ORIGINAL_FEATURES:
    if c not in CAT_COLS_RAW: oX[c] = oX[c].astype("float32")
om = xgb.XGBClassifier(objective="binary:logistic", n_estimators=600, learning_rate=0.05, max_depth=8,
    subsample=0.8, colsample_bytree=0.5, tree_method="hist", enable_categorical=True,
    n_jobs=-1, random_state=0, verbosity=0).fit(oX, oy)
def apply_om(df):
    Xr = df[ORIGINAL_FEATURES].copy()
    for c in CAT_COLS_RAW: Xr[c] = pd.Categorical(Xr[c], categories=_v[c])
    for c in ORIGINAL_FEATURES:
        if c not in CAT_COLS_RAW: Xr[c] = Xr[c].astype("float32")
    p = om.predict_proba(Xr)[:,1].astype("float32")
    return p, np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).astype("float32")
train["orig_proba"], train["orig_logit"] = apply_om(train)
test["orig_proba"], test["orig_logit"] = apply_om(test)
_ce = {"Gender":{"Male":1,"Female":0},"Customer Type":{"Loyal Customer":1,"disloyal Customer":0},
    "Type of Travel":{"Business travel":1,"Personal Travel":0},"Class":{"Business":2,"Eco Plus":1,"Eco":0}}
_enc = orig[ORIGINAL_FEATURES].copy()
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
    df["avg_rating"]=df[RATING_COLS].mean(axis=1); df["min_rating"]=df[RATING_COLS].min(axis=1)
    df["max_rating"]=df[RATING_COLS].max(axis=1); df["rating_std"]=df[RATING_COLS].std(axis=1)
    df["zero_count"]=(df[RATING_COLS]==0).sum(axis=1); df["has_zero"]=(df["zero_count"]>0).astype(int)
    df["age_group"]=pd.cut(df["Age"],bins=[0,18,30,45,60,100],labels=[0,1,2,3,4]).astype(float)
    df["biz_biz"]=((df["Type of Travel"]==1)&(df["Class"]==2)).astype(int)
    df["distance_bin"]=pd.cut(df["Flight Distance"],bins=[0,500,1500,float("inf")],labels=[0,1,2]).astype(float)
    return df

train_p = base_preprocess(train); test_p = base_preprocess(test)
FEATS = [c for c in train_p.columns if c!="satisfaction"]
X_base = train_p[FEATS].reset_index(drop=True); X_te_base = test_p[FEATS].reset_index(drop=True)
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
def add_dp(Xt,Xv,Xte):
    dc="Flight Distance"; meas=RATING_COLS+["Age","Departure Delay in Minutes","Arrival Delay in Minutes"]
    cp=["Gender","Customer Type","Type of Travel","Class"]
    g=Xt.groupby(dc)[meas].agg(["mean","std"]); cnt=Xt[dc].value_counts()
    prop={c:pd.crosstab(Xt[dc],Xt[c],normalize="index") for c in cp}
    dd={c:{"mean":Xt[c].mean(),"std":Xt[c].std()} for c in meas}
    out=[]
    for fr in [Xt,Xv,Xte]:
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
def add_freq(Xt,Xv,Xte):
    specs=[(c,) for c in BASE_ENC]+KEY_PAIRS; out=[Xt.copy(),Xv.copy(),Xte.copy()]
    for cols in specs:
        nm="__".join(c.lower().replace(" ","_").replace("/","_") for c in cols)
        fm=ck(Xt,cols).value_counts(normalize=True)
        for i,fr in enumerate([Xt,Xv,Xte]):
            fq=ck(fr,cols).map(fm).fillna(0).astype("float32")
            out[i][f"freq_{nm}"]=fq; out[i][f"rarity_{nm}"]=-np.log(fq.clip(lower=1.0/len(Xt))).astype("float32")
    return out
def add_om(Xt,Xv,Xte):
    out=[Xt.copy(),Xv.copy(),Xte.copy()]
    for cols,mp in omm.items():
        ft="orig_mean_"+"__".join(c.lower().replace(" ","_") for c in cols)
        for i,fr in enumerate([Xt,Xv,Xte]):
            k=fr[list(cols)].astype(str).apply("|".join,axis=1)
            out[i][ft]=k.map(mp).fillna(oprior).astype("float32")
    return out
def add_te(Xt,Xv,Xte,yt,fold):
    def bk(df):
        ks={c:df[c] for c in BASE_ENC}
        for cols in KEY_PAIRS:
            nm="pair_"+"__".join(c.lower().replace(" ","_") for c in cols)
            ks[nm]=df[cols[0]]*10+df[cols[1]] if all(c in RATING_COLS for c in cols) else ck(df,cols).astype("category")
        ks["distance_250_bin"]=(df["Flight Distance"]//250).astype("float32"); ks["age_5_bin"]=(df["Age"]//5).astype("float32")
        return pd.DataFrame(ks,index=df.index)
    enc=TargetEncoder(target_type="binary",smooth=20.0,cv=5,shuffle=True,random_state=1042+fold)
    tr=enc.fit_transform(bk(Xt),yt); va=enc.transform(bk(Xv)); te=enc.transform(bk(Xte))
    nm=[f"te_{c}" for c in bk(Xt).columns]
    return (pd.concat([Xt,pd.DataFrame(tr.astype("float32"),columns=nm,index=Xt.index)],axis=1),
            pd.concat([Xv,pd.DataFrame(va.astype("float32"),columns=nm,index=Xv.index)],axis=1),
            pd.concat([Xte,pd.DataFrame(te.astype("float32"),columns=nm,index=Xte.index)],axis=1))

MP = dict(device='cuda:0', random_state=3407, n_cv=1, n_refit=0, n_threads=6, verbosity=0,
    val_metric_name='1-auc_ovr', n_epochs=128, batch_size=1024, tabm_k=16, num_emb_type='pwl',
    num_emb_n_bins=32, d_embedding=16, d_block=256, n_blocks=2, dropout=.1, patience=16,
    allow_amp=True, compile_model=False, share_training_batches=False)

skf = StratifiedKFold(10, shuffle=True, random_state=42)
oof = np.zeros(len(X_base)); pred = np.zeros(len(X_te_base))
for fold, (tr, va) in enumerate(skf.split(X_base, y)):
    Xt,Xv,Xte = add_dp(X_base.iloc[tr], X_base.iloc[va], X_te_base)
    Xt,Xv,Xte = add_freq(Xt,Xv,Xte)
    Xt,Xv,Xte = add_om(Xt,Xv,Xte)
    Xt,Xv,Xte = add_te(Xt,Xv,Xte, y.iloc[tr], fold)
    for f in [Xt,Xv,Xte]:
        f.replace([np.inf,-np.inf],0,inplace=True); f.fillna(0,inplace=True)
        f = f.loc[:,~f.columns.duplicated()].astype("float32")
    Xt = Xt[[c for c in Xt.columns if Xt[c].nunique()>1]]
    Xv = Xv[Xt.columns]; Xte = Xte[Xt.columns]
    m = TabM_D_Classifier(**MP)
    m.fit(Xt, y.iloc[tr], X_val=Xv, y_val=y.iloc[va])
    oof[va] = m.predict_proba(Xv)[:,1]; pred += m.predict_proba(Xte)[:,1]/10
    print(f"fold {fold}: TabM AUC={roc_auc_score(y.iloc[va], oof[va]):.5f}", flush=True)
    del m; torch.cuda.empty_cache()

print(f"\nTabM(航线画像) OOF = {roc_auc_score(y, oof):.5f}", flush=True)
np.save("cache/oof_route_tabm.npy", oof); np.save("cache/pred_route_tabm.npy", pred)
