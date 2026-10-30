"""TabPFN 分阶段核查：拟合 / 首次预测 / 后续预测 各阶段耗时与峰值显存。

带显存保护：
- 每个阶段前检查剩余显存，低于安全水位(默认 1.5GB)则跳过并警告，避免逼近上限导致整机断电；
- 捕获 OOM，只报告不崩；
- 阶段间 empty_cache + 短暂停顿，让功耗尖峰回落，避免连续满载触发供电保护。

用法：
    python tabpfn_stages.py                     # 16k 上下文(默认，8GB 卡有风险)
    python tabpfn_stages.py --n-ctx 4096        # 8GB 卡建议先从小上下文起测
    python tabpfn_stages.py --n-ctx 4096 --min-free-gb 2.0
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier


def free_mem_gb():
    """返回 (剩余显存GB, 总显存GB)。"""
    free, total = torch.cuda.mem_get_info()
    return free / 1e9, total / 1e9


def guard(stage, min_free_gb):
    """阶段前水位检查：剩余显存低于阈值则跳过，返回 False。"""
    free, total = free_mem_gb()
    print(f'[{stage}] 剩余显存 {free:.2f} GB / 总 {total:.2f} GB')
    if free < min_free_gb:
        print(f'    [!] 剩余显存不足 {min_free_gb:.2f} GB，跳过「{stage}」——'
              f'避免逼近上限导致整机断电，请用更小 --n-ctx 重试')
        return False
    return True


def is_oom(e):
    return 'out of memory' in str(e).lower()


def main():
    ap = argparse.ArgumentParser(description='TabPFN 分阶段显存/耗时核查（带显存保护）')
    ap.add_argument('--n-ctx', type=int, default=16384, help='上下文行数，默认 16384')
    ap.add_argument('--batch', type=int, default=1024, help='每批预测行数，默认 1024')
    ap.add_argument('--min-free-gb', type=float, default=1.5,
                    help='每阶段前要求的最小剩余显存(GB)，默认 1.5')
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('未检测到 CUDA GPU，无法运行')

    props = torch.cuda.get_device_properties(0)
    free, total = free_mem_gb()
    print(f'GPU: {props.name}  总显存 {total:.2f} GB  当前剩余 {free:.2f} GB')
    if total < 10 and args.n_ctx >= 16384:
        print(f'    [!] 8GB 级显存跑 {args.n_ctx} 上下文 fit_with_cache 有超限/断电风险，'
              f'建议先 --n-ctx 4096 或 8192 起测')

    train = pd.read_csv('data/playground-series-s6e10/train.csv')
    N_CTX, BATCH = args.n_ctx, args.batch
    ctx = train.iloc[:N_CTX].drop(columns=['id', 'satisfaction'])
    y_ctx = train.iloc[:N_CTX]['satisfaction'].to_numpy()
    X_pred = train.iloc[N_CTX:N_CTX + 2 * BATCH].drop(columns=['id', 'satisfaction'])
    y_pred = train.iloc[N_CTX:N_CTX + 2 * BATCH]['satisfaction'].to_numpy()

    clf = TabPFNClassifier(model_path='tabpfn_ckpt/tabpfn-v3.5-20260909.safetensors',
                           n_estimators=1, random_state=0, device='cuda',
                           ignore_pretraining_limits=True, fit_mode='fit_with_cache',
                           kv_cache_precision='int8', inference_precision='autocast')
    free, total = free_mem_gb()
    print(f'模型加载后剩余显存 {free:.2f} GB')

    def gb():
        return torch.cuda.max_memory_allocated() / 1e9

    p1 = p2 = None

    # 1) 拟合
    if guard('拟合', args.min_free_gb):
        try:
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            clf.fit(ctx, y_ctx)
            print(f'拟合        {time.time()-t:6.1f}s  峰值显存 {gb():.2f} GB')
        except RuntimeError as e:
            if is_oom(e):
                print(f'    [!] 拟合 OOM：{args.n_ctx} 上下文超出本机显存，'
                      f'请 --n-ctx 减半或换低显存 fit_mode')
            else:
                raise
        finally:
            torch.cuda.empty_cache()
            time.sleep(1.0)  # 让功耗尖峰回落，避免连续满载断电

    # 2) 首次预测（第一批，触发完整注意力/cache 分配，通常是显存峰值）
    if guard('首次预测', args.min_free_gb):
        try:
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            p1 = clf.predict_proba(X_pred.iloc[:BATCH])[:, 1]
            print(f'首次预测    {time.time()-t:6.1f}s  峰值显存 {gb():.2f} GB')
        except RuntimeError as e:
            if is_oom(e):
                print('    [!] 首次预测 OOM：cache 分配超出显存')
            else:
                raise
        finally:
            torch.cuda.empty_cache()
            time.sleep(1.0)

    # 3) 后续预测（第二批，cache 已热）
    if guard('后续预测', args.min_free_gb):
        try:
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            p2 = clf.predict_proba(X_pred.iloc[BATCH:2 * BATCH])[:, 1]
            print(f'后续预测    {time.time()-t:6.1f}s  峰值显存 {gb():.2f} GB')
        except RuntimeError as e:
            if is_oom(e):
                print('    [!] 后续预测 OOM')
            else:
                raise
        finally:
            torch.cuda.empty_cache()

    # 汇总
    if p1 is not None and p2 is not None:
        p = np.concatenate([p1, p2])
        print(f'AUC（{2 * BATCH} 预测行）={roc_auc_score(y_pred, p):.5f}')
    else:
        print('未完成全部预测阶段，未计算 AUC')


if __name__ == '__main__':
    main()
