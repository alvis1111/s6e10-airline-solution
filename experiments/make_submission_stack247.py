"""落袋：用 stack247_test.npy 生成 0.96187 的提交文件。

stack247_test.npy 是 299844 行 P(satisfaction=1)，test.csv 原始文件顺序。
与 sample_submission.csv 的 id 列逐位对齐后写出。
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

DATA = "data/playground-series-s6e10"
G = "pulled_golem"

test_id = pd.read_csv(f"{DATA}/test.csv", usecols=["id"])
sample = pd.read_csv(f"{DATA}/sample_submission.csv")
s247t = np.load(f"{G}/stack247_test.npy")

print(f"test.csv id      : {len(test_id)} rows, {test_id['id'].nunique()} unique, NaN={test_id['id'].isna().sum()}")
print(f"sample_submission: {len(sample)} rows, cols={list(sample.columns)}")
print(f"stack247_test    : {s247t.shape} {s247t.dtype}, range=[{s247t.min():.6f}, {s247t.max():.6f}], NaN={np.isnan(s247t).sum()}")

assert len(test_id) == len(s247t) == 299844, "row count mismatch"
assert test_id["id"].equals(sample["id"]), "test id order differs from sample_submission id order"
assert np.all((s247t >= 0) & (s247t <= 1)), "prediction out of [0,1]"

sub = pd.DataFrame({"id": test_id["id"], "satisfaction": s247t})
out = "submission_stack247.csv"
sub.to_csv(out, index=False)

digest = hashlib.sha256(sub["satisfaction"].to_numpy().tobytes()).hexdigest()
print(f"\nwrote {out}  ({len(sub)} rows)")
print(f"satisfaction 字节序列 sha256 前 16 位 = {digest[:16]}  (README 记载提交的 sha256 以 ac2f1c51d067615e 开头)")
