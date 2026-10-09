"""生成 63 成员（53 + 10 sachith7）logistic 融合提交。"""
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("data/playground-series-s6e10/train.csv")
test = pd.read_csv("data/playground-series-s6e10/test.csv")
sub = pd.read_csv("data/playground-series-s6e10/sample_submission.csv")
y = train["satisfaction"].to_numpy(np.int64)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

MEM = r"C:/Users/GEM07/.cache/kagglehub/notebooks/goodpjw2008/s6e10-tabpfn-route-categories-lb-0-96160/output/versions/1"
mo = pd.read_csv(MEM + "/oof_members.csv").set_index("id").reindex(train["id"])
mt = pd.read_csv(MEM + "/test_members.csv").set_index("id").reindex(test["id"])
z = np.load("pulled_busy53/oof_ours.npz", allow_pickle=False)
own_names = ["lgbm", "lgbm_xt", "cat", "xgb", "realmlp"]

# OOF + test（保持同一顺序）
X53 = np.column_stack([logit(mo[c].to_numpy()) for c in mo.columns] + [logit(z[n]) for n in own_names])
T53 = np.column_stack([logit(mt[c].to_numpy()) for c in mt.columns] + [logit(z["test_" + n]) for n in own_names])

SA = r"C:/Users/GEM07/.cache/kagglehub/datasets/sachith7/s6e10-stack-oof-predictions/versions/2"
sa_files = sorted(glob.glob(SA + "/*_k10*.npz"))
Xsa = np.column_stack([logit(np.load(f, allow_pickle=False)["oof"]) for f in sa_files])
Tsa = np.column_stack([logit(np.load(f, allow_pickle=False)["test"]) for f in sa_files])

X = np.column_stack([X53, Xsa])
T = np.column_stack([T53, Tsa])
print(f"成员 {X.shape[1]}，OOF {X.shape}，test {T.shape}", flush=True)

m = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
logit_test = m.decision_function(T)
sub["satisfaction"] = 1 / (1 + np.exp(-logit_test))
sub.to_csv("submission_63_sachith7.csv", index=False)
print("saved -> submission_63_sachith7.csv", flush=True)
