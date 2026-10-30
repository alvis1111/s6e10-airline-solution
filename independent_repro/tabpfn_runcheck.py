"""TabPFN 运行检查（计划第1步）：16,384 上下文 + 2,048 预测，用公开方案的配置。

goodpjw2008 train_tabpfn.py 的配置：n_estimators=1、kv_cache_precision=int8、
inference_precision=autocast、ignore_pretraining_limits=True、fit_mode=fit_with_cache、batch 分块预测。
"""
import time
import numpy as np
import pandas as pd
import tabpfn
import torch
from tabpfn import TabPFNClassifier
from sklearn.metrics import roc_auc_score

train = pd.read_csv('data/playground-series-s6e10/train.csv')
N_CTX, N_PRED, BATCH = 16384, 2048, 1024
ctx = train.iloc[:N_CTX].drop(columns=['id', 'satisfaction'])
y_ctx = train.iloc[:N_CTX]['satisfaction'].to_numpy()
X_pred = train.iloc[N_CTX:N_CTX + N_PRED].drop(columns=['id', 'satisfaction'])
y_pred = train.iloc[N_CTX:N_CTX + N_PRED]['satisfaction'].to_numpy()

print('tabpfn', getattr(tabpfn, '__version__', '?'))
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())

t0 = time.time()
clf = TabPFNClassifier(model_path='tabpfn_ckpt/tabpfn-v3.5-20260909.safetensors',
                       n_estimators=1, random_state=0, device='cuda',
                       ignore_pretraining_limits=True, fit_mode='fit_with_cache',
                       kv_cache_precision='int8', inference_precision='autocast')
t_load = time.time() - t0
clf.fit(ctx, y_ctx)
t_fit = time.time() - t0

p = np.zeros(N_PRED, dtype=np.float64)
t_pred0 = time.time()
for i in range(0, N_PRED, BATCH):
    p[i:i + BATCH] = clf.predict_proba(X_pred.iloc[i:i + BATCH])[:, 1]
dt = time.time() - t0
print(f'加载 {t_load:.1f}s，fit {t_fit - t_load:.1f}s，predict {dt - t_fit:.1f}s，总计 {dt:.1f}s')
print(f'AUC={roc_auc_score(y_pred, p):.5f}')
if torch.cuda.is_available():
    print(f'显存峰值 {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
