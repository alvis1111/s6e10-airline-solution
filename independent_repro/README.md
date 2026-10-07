# 独立复现（漏掉的杠杆）

本目录是对复盘认定的「漏掉的杠杆」逐一**独立复现**的脚本与结果。最终把自有成绩从 **0.96100 推到 0.96126**（公榜）。

完整思路与教训见 [S6E10独立复现总结.md](./S6E10独立复现总结.md)。

## 最终成绩（自有）

| 阶段 | LB |
|---|---:|
| 自有融合（30/30/40） | 0.96100 |
| +50% 精简 RealMLP | 0.96121 |
| +10% 辅助 RealMLP（Codex 会话） | 0.96125 |
| +10% 深层 XGBoost | **0.96126** |

抄公开（Busy53）0.96169；榜一 0.96177。

## 候选结论速览

| 候选 | 结论 |
|---|---|
| TabPFN 32k 降级版 | 互补性门未通过（完整 560k context 需 GB10/48GB，8GB 只跑 32k） |
| 原始数据 lookup（18 列） | fold 0 负，停 |
| digit（整数数字位） | 严格版两折负，停 |
| 精简 RealMLP（Demidov 配置） | ✅ 最大赢家，+0.00021 LB |
| 深层 LightGBM | 冗余于 RealMLP，停 |
| 深层 XGBoost | 贴线正，+0.00001 LB |
| 深度 8 CatBoost | fold 0 负，停 |
| 网格搜索 5 路权重 | OOF 高估、公榜反转（0.96116 < 0.96126） |

## 脚本索引

通用工具：
- `delong.py` — 配对 DeLong 检验（AUC 差值显著性）。

RealMLP（精简版，Demidov 配置，来源 busyaprime/yekenot）：
- `realmlp_repro_lean.py` — 两折初筛（单折可传参）
- `realmlp_full.py` — 10 折 + 提交
- `realmlp_blend_5050.py` — 50/50 融合提交（→ 0.96121）
- `realmlp_seed2.py` — 第二种子两折（无增益）
- `realmlp_delong.py` — 两折配对 DeLong

深层 GBDT（busyaprime 的 lgbm/xgb/cat 三腿，同一套 org/route/digit/freq + 全 21+10 auto TE）：
- `lgbm_deep_repro.py` / `lgbm_deep_5050_test.py` — 深层 LightGBM
- `xgb_deep_repro.py` / `xgb_deep_full.py` / `xgb_deep_delong.py` / `xgb_deep_delong2.py` — 深层 XGBoost（proper 嵌套：内层选轮数 + 外层重拟合）
- `cat_deep_repro.py` — 深度 8 CatBoost（GPU）

其他候选：
- `orig_lookup.py` / `lookup_paired.py` — 原始数据逐列 lookup
- `digit_paired.py` / `digit_delong.py` — 整数数字位
- `tabpfn_repro.py` / `complementarity_screen.py` — TabPFN 32k
- `blend_ours.py` / `blend_ours_submit.py` — 多腿网格搜索（OOF 搜权会高估，见教训）

提交（`submissions/`）：
- `submission_realmlp_5050.csv` — 0.96121
- `submission_xgb_deep.csv` — **0.96126（当前自有最佳）**
- `submission_blend_ours_5leg.csv` — 0.96116（网格搜权公榜反转）

## 关键教训

1. **报告结论别过度断言**：「未显示增益 / 停止当前配置」≠「判死 / 无增益 / 特征饱和坐实」。
2. **测量增益用配对 DeLong/bootstrap**，别只看 OOF 差值。
3. **特征消融要 proper 嵌套**（内层选轮数用独立特征/TE，不与外层共享），否则 val 早停泄漏、增益偏乐观。
4. **别在 full-OOF 上搜权重**（网格搜索 OOF 0.961716 → 公榜 0.96116 反转；手调 50/50 更稳）。
5. **n_ens 已覆盖集成收益**（RealMLP 第二种子无增益）。

## 依赖说明

脚本含本机绝对路径，克隆后需改：
- 竞赛数据：`data/playground-series-s6e10/`
- 原始数据：`C:/Users/GEM07/.cache/kagglehub/datasets/teejmahal20/airline-passenger-satisfaction/versions/1/`
- 自有融合缓存：`cache/oof_route_aux_l.npy`、`oof_route_aux_x.npy`、`oof_route_tabm.npy`、`oof_realmlp_lean.npy` 等
- 0.96126 基线：`C:/Users/GEM07/Documents/Codex/.../outputs/deep_xgb_confirmation/full_096126/predictions.npz`（含 oof_candidate/pred_candidate）
