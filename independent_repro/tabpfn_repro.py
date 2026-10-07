"""TabPFN-3.5 独立复现（32k 降级版）：raw 与 raw+catfd 两个对照，自生成 OOF + 提交。

参考 goodpjw2008 的公开 train_tabpfn.py 配置（注明来源，Apache-2.0），唯一差异：
- 参考机是 GB10 / 121GB 统一内存，每折全量 ~560k 行 context；本机 8GB 只能跑 32k context（~5.7%）。
- 因此这是「降级复现」，单腿分数预期低于公开 full-context member（full_raw 0.96073 / catfd 0.96092）。

配置对齐参考脚本：
  n_estimators=1、kv_cache_precision=int8、inference_precision=autocast、
  ignore_pretraining_limits=True、fit_mode=fit_with_cache、memory_saving_mode=auto、
  分类特征 4 列（catfd 再加 Flight Distance），Arrival Delay 填 -1。

用法：
  python tabpfn_repro.py --smoke          # 冒烟：32k 拟合 + 2048 行预测，验证配置（~40s）
  python tabpfn_repro.py --folds 1        # 只跑 fold 0（互补性初筛用，~13min/变体）
  python tabpfn_repro.py                  # 完整 5 折 × 2 变体，逐折 checkpoint 可续跑
  python tabpfn_repro.py --variants raw   # 只跑 raw
"""
import argparse
import hashlib
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier

warnings.filterwarnings("ignore")

DATA = "data/playground-series-s6e10"
CKPT = "tabpfn_ckpt/tabpfn-v3.5-20260909.safetensors"
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
PART = os.path.join("cache", "partial")
OUT = "tabpfn_repro"
FOLD_SCHEME = "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)"


def _file_hash(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def config_sig(variant, ctx, seed, nfolds, cat_idx, feats, data_h, model_h):
    """配置签名：任一成分（ctx/seed/折方案/特征顺序/数据/权重/版本）变了，缓存名就变，避免混用旧 fold。"""
    s = "|".join([
        "v1", FOLD_SCHEME, f"variant={variant}", f"ctx={ctx}", f"seed={seed}", f"nfolds={nfolds}",
        f"cat_idx={','.join(map(str, cat_idx))}", f"feats={','.join(feats)}",
        f"data={data_h}", f"model={model_h}",
    ])
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def predict(clf, Xq, batch):
    out = np.empty(len(Xq), dtype=np.float32)
    for i in range(0, len(Xq), batch):
        out[i:i + batch] = clf.predict_proba(Xq[i:i + batch])[:, 1]
    return out


def make_clf(cat_idx, seed):
    return TabPFNClassifier(
        model_path=CKPT, n_estimators=1, random_state=seed, device="cuda",
        ignore_pretraining_limits=True, fit_mode="fit_with_cache",
        kv_cache_precision="int8", inference_precision="autocast",
        categorical_features_indices=cat_idx, memory_saving_mode="auto")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="raw,catfd")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mem-frac", type=float, default=0.85)
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
        free, total = torch.cuda.mem_get_info()
        print(f"GPU: {torch.cuda.get_device_name(0)}  总 {total/1e9:.2f}GB  剩余 {free/1e9:.2f}GB", flush=True)
    else:
        raise SystemExit("未检测到 CUDA GPU")

    tr = pd.read_csv(f"{DATA}/train.csv")
    te = pd.read_csv(f"{DATA}/test.csv")
    y = tr["satisfaction"].to_numpy(np.int64)
    FEATS = [c for c in tr.columns if c not in ("id", "satisfaction")]
    full = pd.concat([tr[FEATS], te[FEATS]], ignore_index=True)
    for c in CATS:
        full[c] = full[c].astype("category").cat.codes
    full["Arrival Delay in Minutes"] = full["Arrival Delay in Minutes"].fillna(-1)
    X = full.values.astype(np.float32)
    Xtr, Xte = X[:len(tr)], X[len(tr):]
    cat_idx_base = [full.columns.get_loc(c) for c in CATS]
    fd_idx = full.columns.get_loc("Flight Distance")

    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(Xtr, y))
    data_h = _file_hash(f"{DATA}/train.csv")[:12] + _file_hash(f"{DATA}/test.csv")[:12]
    model_h = _file_hash(CKPT)[:12]
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PART, exist_ok=True)

    for variant in args.variants.split(","):
        variant = variant.strip()
        cat_idx = cat_idx_base + ([fd_idx] if variant == "catfd" else [])
        sig = config_sig(variant, args.ctx, args.seed, args.folds, cat_idx, FEATS, data_h, model_h)
        print(f"\n=== variant={variant}  cat_idx={cat_idx} ===", flush=True)

        if args.smoke:
            a = folds[0][0]
            a = np.random.RandomState(args.seed).choice(a, args.ctx, replace=False)
            clf = make_clf(cat_idx, args.seed)
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            clf.fit(Xtr[a], y[a])
            p = predict(clf, Xtr[folds[0][1]][:2048], args.batch)
            auc = roc_auc_score(y[folds[0][1]][:2048], p)
            print(f"[smoke] ctx={len(a)}  AUC(2048)={auc:.5f}  "
                  f"耗时 {time.time()-t:.0f}s  峰值 {torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
            del clf
            torch.cuda.empty_cache()
            continue

        oof = np.zeros(len(tr), dtype=np.float32)
        pred = np.zeros(len(te), dtype=np.float32)
        for f, (a, b) in enumerate(folds[:args.folds]):
            part = os.path.join(PART, f"tabpfn_repro_{variant}_f{f}_{sig}.npz")
            if os.path.exists(part):
                d = np.load(part)
                oof[b] = d["oof"]
                pred += d["test"] / args.folds
                print(f"  fold {f} restored  auc {roc_auc_score(y[b], oof[b]):.6f}", flush=True)
                continue
            if args.ctx > 0:
                a = np.random.RandomState(args.seed).choice(a, args.ctx, replace=False)
            # args.ctx == 0：全量 context（a 保持整个训练折），与参考脚本 ctx=0 语义一致；8GB 会 OOM
            clf = make_clf(cat_idx, args.seed)
            t = time.time()
            clf.fit(Xtr[a], y[a])
            oof[b] = predict(clf, Xtr[b], args.batch)
            pt = predict(clf, Xte, args.batch)
            pred += pt / args.folds
            np.savez(part, oof=oof[b], test=pt)
            print(f"  fold {f}: ctx={len(a)}  auc {roc_auc_score(y[b], oof[b]):.6f}  "
                  f"耗时 {time.time()-t:.0f}s  峰值 {torch.cuda.max_memory_allocated()/2**30:.2f}GB", flush=True)
            del clf
            torch.cuda.empty_cache()

        if args.folds == 5:
            auc = roc_auc_score(y, oof)
            print(f"variant={variant}  OOF AUC = {auc:.6f}", flush=True)
            pd.DataFrame({"id": tr["id"], "pred": oof}).to_csv(f"{OUT}/oof_tabpfn_{variant}.csv", index=False)
            pd.DataFrame({"id": te["id"], "pred": pred}).to_csv(f"{OUT}/test_tabpfn_{variant}.csv", index=False)
            np.save(f"{OUT}/oof_tabpfn_{variant}.npy", oof)
            np.save(f"{OUT}/test_tabpfn_{variant}.npy", pred)
            print(f"saved -> {OUT}/oof_tabpfn_{variant}.csv / test_tabpfn_{variant}.csv", flush=True)


if __name__ == "__main__":
    main()
