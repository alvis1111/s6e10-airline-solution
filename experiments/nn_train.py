"""S6E10 — 神经网络 (优化版): GPU手动分批, 去掉DataLoader开销. 5折CV."""
import os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA = "data/playground-series-s6e10"
TARGET = "satisfaction"
N_FOLDS = 5
SEEDS = [42, 202]
OUT = "cache"
os.makedirs(OUT, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device =", device, flush=True)

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

train = pd.read_csv(f"{DATA}/train.csv")
test = pd.read_csv(f"{DATA}/test.csv")
y = train[TARGET].astype(int).values

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
SERVICE_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Cleanliness",
]
NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"] + SERVICE_COLS

def build_features(df):
    df = df.copy()
    df["service_sum"] = df[SERVICE_COLS].sum(axis=1)
    df["service_mean"] = df[SERVICE_COLS].mean(axis=1)
    df["service_std"] = df[SERVICE_COLS].std(axis=1)
    df["total_delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    df["is_delayed"] = (df["total_delay"] > 0).astype(int)
    df["travel_x_class"] = df["Type of Travel"] + "_" + df["Class"]
    df["travel_x_customer"] = df["Type of Travel"] + "_" + df["Customer Type"]
    return df

train = build_features(train)
test = build_features(test)
NUM_COLS = NUM_COLS + ["service_sum", "service_mean", "service_std", "total_delay", "is_delayed"]
CAT_COLS = CAT_COLS + ["travel_x_class", "travel_x_customer"]

for df in (train, test):
    for c in NUM_COLS:
        if df[c].isna().any():
            df[c] = df[c].fillna(0)

encoders = {}
for c in CAT_COLS:
    le = LabelEncoder()
    le.fit(pd.concat([train[c], test[c]]).astype(str))
    train[c] = le.transform(train[c].astype(str))
    test[c] = le.transform(test[c].astype(str))
    encoders[c] = le
cardinalities = [len(encoders[c].classes_) for c in CAT_COLS]

scaler = StandardScaler()
scaler.fit(train[NUM_COLS].astype("float32"))
X_num_tr = scaler.transform(train[NUM_COLS].astype("float32")).astype("float32")
X_num_te = scaler.transform(test[NUM_COLS].astype("float32")).astype("float32")
X_cat_tr = train[CAT_COLS].astype("int64").values
X_cat_te = test[CAT_COLS].astype("int64").values

Xn_te_g = torch.tensor(X_num_te, device=device)
Xc_te_g = torch.tensor(X_cat_te, device=device)

class TabularNet(nn.Module):
    def __init__(self, num_dim, card, emb_dim=8, hidden=(512, 256, 128), dropout=0.3):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim) for c in card])
        in_dim = num_dim + len(card) * emb_dim
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, xn, xc):
        embs = [emb(xc[:, i]) for i, emb in enumerate(self.embs)]
        return self.mlp(torch.cat([xn] + embs, dim=1)).squeeze(1)

@torch.no_grad()
def predict(model, xn, xc, bs=65536):
    model.eval()
    outs = [torch.sigmoid(model(xn[i:i+bs], xc[i:i+bs])).cpu().numpy() for i in range(0, len(xn), bs)]
    return np.concatenate(outs)

def train_fold(Xn_tr, Xc_tr, y_tr, Xn_va, Xc_va, y_va, seed):
    seed_everything(seed)
    model = TabularNet(Xn_tr.shape[1], cardinalities).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    loss_fn = nn.BCEWithLogitsLoss()

    Xn_tr_g = torch.tensor(Xn_tr, device=device)
    Xc_tr_g = torch.tensor(Xc_tr, device=device)
    y_tr_g = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xn_va_g = torch.tensor(Xn_va, device=device)
    Xc_va_g = torch.tensor(Xc_va, device=device)
    y_va_np = y_va
    n = len(Xn_tr_g)
    BATCH = 16384

    best_auc, best_state, patience = 0.0, None, 0
    for epoch in range(60):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xn_tr_g[idx], Xc_tr_g[idx]), y_tr_g[idx])
            loss.backward()
            opt.step()
        sched.step()

        pred_va = predict(model, Xn_va_g, Xc_va_g)
        auc = roc_auc_score(y_va_np, pred_va)
        if auc > best_auc:
            best_auc, best_state, patience = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 10:
                break
        if epoch == 0 or epoch % 5 == 0:
            print(f"    epoch {epoch:2d}: auc={auc:.5f}  ({time.time()-t0:.1f}s)", flush=True)

    model.load_state_dict(best_state)
    return predict(model, Xn_va_g, Xc_va_g), predict(model, Xn_te_g, Xc_te_g), best_auc

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
for seed in SEEDS:
    print(f"\n=== NN seed {seed} ===", flush=True)
    oof = np.zeros(len(train)); pred = np.zeros(len(test))
    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        oof_va, tp, ba = train_fold(
            X_num_tr[tr_idx], X_cat_tr[tr_idx], y[tr_idx],
            X_num_tr[va_idx], X_cat_tr[va_idx], y[va_idx], seed)
        oof[va_idx] = oof_va; pred += tp / N_FOLDS
        print(f"  fold {fold}: val_auc={ba:.5f}", flush=True)
    np.save(f"{OUT}/oof_nn_{seed}.npy", oof)
    np.save(f"{OUT}/pred_nn_{seed}.npy", pred)
    print(f"  NN seed {seed} OOF AUC = {roc_auc_score(y, oof):.5f}", flush=True)
