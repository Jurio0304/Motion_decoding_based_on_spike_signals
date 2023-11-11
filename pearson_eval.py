import math
import sys
import time
import numpy as np
from scipy.stats import pearsonr


def pearson(vector1, vector2):
    n = len(vector1)
    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))
    # 平方和
    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    # 乘积和
    p_sum = sum([vector1[i] * vector2[i] for i in range(n)])
    # 分子&分母, numerator and denominator
    num = p_sum - (sum1 * sum2 / n)
    den = math.sqrt(abs((sum1_pow - pow(sum1, 2) / n) * (sum2_pow - pow(sum2, 2) / n)))
    if den == 0:
        return 0.0
    return num / den


def cal_CorrCoeff(targets, outputs):
    """
    cc
    :param outputs:
    :param targets:
    :return:
    """
    c_array = np.zeros(outputs.shape[1])
    for i in range(outputs.shape[1]):
        c_array[i] = np.corrcoef(outputs[:, i], targets[:, i])[0][1]
    return np.mean(c_array)


def pearson_mel(mel1, mel2):
    """
    计算基于Mel频段的平均Pearson相关系数
    :param mel1:
    :param mel2:
    :return:
    """
    p_mel = []
    for i in range(0, len(mel1[0, :])):
        # p_mel.append(pearson(mel1[:, i], mel2[:, i]))
        p_mel.append(pearsonr(mel1[:, i], mel2[:, i])[0])

    return np.mean(p_mel), np.std(p_mel)


def pearson_mel_fig(mel1, mel2):
    """
    分别基于Mel频段和时间箱计算平均Pearson相关系数p_mel, p_time
    p = λ * p_mel + (1-λ) * p_time
    :param mel1:
    :param mel2:
    :return:
    """
    lam = 0.5
    p_mel = []
    p_time = []
    for i in range(0, len(mel1[0, :])):
        p_mel.append(pearson(mel1[:, i], mel2[:, i]))
    for j in range(0, len(mel1[:, 0])):
        p_time.append(pearson(mel1[j, :], mel2[j, :]))

    p = lam * np.mean(p_mel) + (1-lam) * np.mean(p_time)

    return p


if __name__ == '__main__':
    np.random.seed(0)
    size = 500
    v1 = np.random.normal(0, 1, size)
    v2 = np.random.normal(0, 10, size)

    time_start = time.time()
    r1_low = pearson(v1, v1 + v1)
    r1_high = pearson(v1, v1 + v2)
    time_end = time.time()
    print(" r1_low =", r1_low, '\n', "r1_high =", r1_high, '\n', "cost =", time_end - time_start, "s")

    time_start = time.time()
    p2, r2 = pearsonr(v1, v1 + v1)
    r2_high = pearsonr(v1, v1 + v2)
    time_end = time.time()
    print(" low(r2, p2) =", p2, '\n', "high(r2, p2) =", r2_high, '\n', "cost =", time_end - time_start, "s")

    sys.exit(0)
