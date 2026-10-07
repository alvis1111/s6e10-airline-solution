# Kaggle Playground S6E10 — Predicting Airline Satisfaction

二分类（ROC-AUC）。本仓库主体是 **0.96100** 的三路融合方案（从 0.95862 起，+0.00238）。后续「漏掉的杠杆」独立复现把**自有成绩推到 0.96126**，抄公开 Busy53 到 0.96169（榜一 0.96177）——见 `independent_repro/`。

## 最终方案（三路融合）

| 路线 | OOF | 说明 |
|---|---|---|
| 航线画像 GBDT（LGBM+XGB 3种子，10折） | 0.96123 | 精确距离分组 + 原始数据特征 |
| context_boost（条件 TE + 频率） | 0.96114 | 并行会话的条件编码路线 |
| 航线画像 TabM（pytabkit，NN） | 0.96077 | NN 补多样性 |

权重 **30% / 30% / 40%**，10 折外层验证（每模型 90% 训练数据）。

## 核心教训：正确目标编码才是关键

早期用「手工平滑 m=20、只编 4 个类别列」→ 误判「目标编码无用」，卡死在 0.959。

正确做法：**sklearn `TargetEncoder`，`smooth='auto'`（或 20/100）、`cv=5` 内层交叉拟合、对全部 21 列（含数值列转字符串）编码** → 单 LGBM 直接 0.96029（+0.0013）。

> **教训：一个技术试一次没效果，先怀疑实现是否到位，别急着否定技术本身。**

## 有效技术栈（按发现顺序逐层叠加）

1. 正确目标编码（smooth auto，全列，内层 cv=5）→ +0.0013
2. 条件特征：服务评分 × traveler_segment（Travel|Customer|Class 三阶）
3. 数值分组 TE：Age/距离 × 出行类型/客户类型/舱位（smooth 20）
4. 频率编码：`log1p(1e6 × count / fit_rows)`
5. 航线画像：按精确 Flight Distance 分组，算每航线评分的均值/标准差/残差 + 类别占比
6. 原始数据特征：真实 129k 数据训 XGBoost，其 `orig_proba`/`orig_logit` 当特征
7. 精确距离 TE（`te_Flight Distance` 排名第一，不是分桶）

## 仓库结构

```
├── src/                     # 核心（最终方案）
│   ├── route_profile.py      # 航线画像 + 原始数据 + 精确TE + 10折 GBDT（核心）
│   ├── tabm_route.py         # 航线画像 TabM（NN 第三条腿）
│   ├── te_cond_freq.py       # 条件TE + 频率 GBDT 集成
│   ├── fusion.py             # 早期 ens/tree/TabM 融合（非最终 30/30/40）
│   └── te_proper.py          # 正确 TE 复现（教学）
├── experiments/             # 25 个实验/死路脚本（早期 GBDT、NN、特征搜索、伪标签等）
├── notebooks/
│   └── s6e10_solution.ipynb  # 讲解版 notebook（EDA + 洞察 + 基线）
├── data/                    # 数据下载说明（数据不提交）
├── independent_repro/       # 「漏掉的杠杆」独立复现（0.96100 → 0.96126），见其 README
└── submission_final_0.96100.csv  # 最终提交
```

## 复现

```bash
pip install -r requirements.txt
# 数据放 data/playground-series-s6e10/（train.csv/test.csv/sample_submission.csv）
python src/route_profile.py        # 航线画像 GBDT（需 CPU 数十分钟）
python src/tabm_route.py           # 航线画像 TabM（需 CUDA GPU）
python src/fusion.py               # 早期 ens/tree/TabM 融合（非最终 30/30/40）
```

> ⚠️ 复现说明：`src/fusion.py` 是**早期**的「我的集成 + 并行树 + 反推 TabM」2-way/3-way 融合，
> **不是**最终 30/30/40 的三路融合。最终 0.96100 = rank-blend 三腿：`route_profile.py`（航线画像 GBDT）
> + **context_boost（条件 TE + 频率，并行会话产物，未随本仓库提交）** + `tabm_route.py`（TabM）。
> 三腿权重 30/30/40 是简单的 rank 平均，融合脚本未单列。

## 已验证无效（别重走）

GBDT 超参调优、弱 TE、特征穷举（聚合/交互/0标志/3-way）、NN（MLP/embedding/NN+TE 全 0.957、裸 TabM 0.957（航线画像 TabM 0.96077 是三路融合的一腿）；RealMLP 默认配置 0.957 是死路，但 Demidov 调参后 0.9613 是后续独立复现的最大赢家，见 `independent_repro/`）、AutoGluon、伪标签、分组模型、真实数据加权训练、删噪声特征、评分×年龄/航程交互。

## 环境

RTX 5060 Laptop(8GB)。torch 2.11+cu128、pytabkit 1.7.3、AutoGluon 1.6.3（会把 pandas 降到 2.3.3）。

## 坑

- XGBoost 3.3 早停写构造参数 `early_stopping_rounds`（不是 fit/callbacks）
- 70 万行 DataLoader 慢，NN 要 GPU 手动分批
- TabM 10 折 GPU 上超 2 小时（后台会超时被杀）
- pandas `.map` 在 category 列返回 category dtype，要显式 `.astype(float)`
