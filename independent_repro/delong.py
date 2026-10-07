"""配对 DeLong 检验：两个模型在同一测试集上的 AUC 差值显著性。

用于 step 3 互补性初筛/确认——判断「自有融合 + w·TabPFN」相对「自有融合」的 AUC 增益
是否统计显著（95% 区间下界 > 0），而不是只盯 OOF 差值。

实现：标准 DeLong 非参数方差估计（向量化，O(n log n)），placement 用 np.searchsorted
在排序后的对方类分数上二分，含同分 0.5 处理。返回 AUC 差值与双侧 95% 正态区间。
"""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def _placements(scores, labels):
    """返回正类的 S10（长度 n_pos）与负类的 S01（长度 n_neg）。"""
    pos = labels == 1
    neg = labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("需要同时包含正负两类样本")

    s_neg = np.sort(scores[neg])
    s_pos = np.sort(scores[pos])

    # 正样本 S10 = P(负类分数 < p_i) + 0.5·P(负类分数 == p_i)
    lt = np.searchsorted(s_neg, scores[pos], side="left")
    le = np.searchsorted(s_neg, scores[pos], side="right")
    S10 = (lt + 0.5 * (le - lt)) / n_neg

    # 负样本 S01 = P(正类分数 > p_j) + 0.5·P(正类分数 == p_j)
    lt = np.searchsorted(s_pos, scores[neg], side="left")
    le = np.searchsorted(s_pos, scores[neg], side="right")
    S01 = (n_pos - le + 0.5 * (le - lt)) / n_pos

    return S10, S01, n_pos, n_neg


def _auc_cov(a10, a01, b10, b01, n_pos, n_neg):
    """Cov(AUC_a, AUC_b)（无偏，n(n-1) 分母）。每个 S 都以自身均值(=AUC)居中。"""
    c10 = np.sum((a10 - a10.mean()) * (b10 - b10.mean())) / (n_pos * (n_pos - 1))
    c01 = np.sum((a01 - a01.mean()) * (b01 - b01.mean())) / (n_neg * (n_neg - 1))
    return c10 + c01


def delong_ci(y, p_base, p_new, alpha=0.05):
    """AUC_new - AUC_base 的 (1-alpha) 双侧置信区间与 p 值。

    返回 (auc_base, auc_new, delta, lower, upper, p_value)。
    """
    y = np.asarray(y, dtype=np.int64)
    p_base = np.asarray(p_base, dtype=np.float64)
    p_new = np.asarray(p_new, dtype=np.float64)

    auc_base = roc_auc_score(y, p_base)
    auc_new = roc_auc_score(y, p_new)

    S10_b, S01_b, n_pos, n_neg = _placements(p_base, y)
    S10_n, S01_n, _, _ = _placements(p_new, y)

    var_b = _auc_cov(S10_b, S01_b, S10_b, S01_b, n_pos, n_neg)
    var_n = _auc_cov(S10_n, S01_n, S10_n, S01_n, n_pos, n_neg)
    cov_bn = _auc_cov(S10_b, S01_b, S10_n, S01_n, n_pos, n_neg)

    delta = auc_new - auc_base
    se = np.sqrt(max(var_b + var_n - 2 * cov_bn, 0.0))

    zc = norm.ppf(1 - alpha / 2)
    lower, upper = delta - zc * se, delta + zc * se
    p_value = 2 * (1 - norm.cdf(abs(delta / se))) if se > 0 else 1.0

    return auc_base, auc_new, delta, lower, upper, p_value


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    n = 50000
    y = rng.randint(0, 2, n)
    sig = rng.randn(n) + 1.5 * y
    p_base = 1 / (1 + np.exp(-sig))                       # AUC ~0.75
    p_new = p_base + 0.015 * rng.randn(n)                 # 加噪声：应略降、区间可能跨 0
    print("identical:", delong_ci(y, p_base, p_base))
    print("noisy:    ", delong_ci(y, p_base, np.clip(p_new, 0, 1)))
    # 加真实信号（小权重）应显著为正
    p_plus = p_base + 0.10 * y                            # 直接注入标签，应显著正
    print("signal:   ", delong_ci(y, p_base, np.clip(p_plus, 0, 1)))
