import numpy as np


def lastNonZero(arr, axis, invalid_val=-1):
    """
    计算arr中每一行最后一个非零数的坐标
    :param arr:
    :param axis:
    :param invalid_val:
    :return: 每一行最后一个非零元素坐标的集合
    """
    mask = arr != 0  # 返回arr.shape大小的bool数组
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1  # 按指定轴翻转数组中的元素
    # 如果最后一个元素不为0或全为0则np.flip(mask, axis=axis).argmax(axis=axis)等于0
    yAxis = np.where(mask.any(axis=axis), val,
                     invalid_val)  # np.any()任意元素为true即输出true#三个参数np.where(cond,x,y)：满足条件（cond）输出x，不满足输出y;cond为bool矩阵
    # 数组全为0则输出-1，不全为0 则输出val，则yaxis为该数组中最后一个不为0的位置索引
    xAxis = np.arange(arr.shape[0], dtype=np.int64)
    xRet = xAxis[yAxis >= 0]
    yRet = yAxis[yAxis >= 0]
    return xRet, yRet


def calEndTimeLB(CT_ope, pt_ope_min, pt_ope_mean):
    """
    计算工序的最大完工时间表，该表由两种表格合并，一种是根据pt_ope_min，计算方法见calEndTimeLBm，另一种与calEndTimeLBm类似，将最小换成平均
    :param CT_ope:
    :param pt_ope_min:
    :param pt_ope_mean:
    :return:
    """
    x, y = lastNonZero(CT_ope, 1, invalid_val=-1)
    pt_ope_min[np.where(CT_ope != 0)] = 0
    pt_ope_mean[np.where(CT_ope != 0)] = 0
    pt_ope_min[x, y] = CT_ope[x, y]
    pt_ope_mean[x, y] = CT_ope[x, y]
    temp20 = np.cumsum(pt_ope_min, axis=1)
    temp21 = np.cumsum(pt_ope_mean, axis=1)  # cumsum按行依次累加
    temp20[np.where(CT_ope != 0)] = 0
    temp21[np.where(CT_ope != 0)] = 0
    temp2 = np.concatenate(
        (temp20.reshape(temp20.shape[0], temp20.shape[1], 1), temp21.reshape(temp20.shape[0], temp20.shape[1], 1)), -1)
    ret = CT_ope.reshape(CT_ope.shape[0], CT_ope.shape[1], 1) + temp2
    return ret


def calEndTimeLBm(CT_ope, pt_ope_min):
    """
    计算工序的最大完工时间表
    工序完工时间表：未加工工序的完工时间 = 该工序前一个工序的完工时间 + 这个工序的最小加工时间
    已加工工序 = 该工序的最大完工时间
    :param CT_ope: 工序的完工时间，加工工序的最小加工时间
    :param pt_ope_min:
    :return: 完工时间表
    """
    x, y = lastNonZero(CT_ope, 1, invalid_val=-1)
    pt_ope_min[np.where(CT_ope != 0)] = 0  # 将已经调度的工序在pt_ope_min对应位置设置为0
    # (temp1 != 0)->已经调度过的工序，dur_cp[np.where(temp1 != 0)] = 0将dur_cp中已经调度过的工序的时间设置为0
    pt_ope_min[x, y] = CT_ope[x, y]  # 将已加工工序的pt_ope_min设置为该工序的完工时间
    temp20 = np.cumsum(pt_ope_min, axis=1)  # pt_ope_min每个工序加工时间 加上 前面已完工工序的完工时间
    temp20[np.where(CT_ope != 0)] = 0  # 将CT_ope不为零的位置的值设置为0
    ret = CT_ope + temp20  # 合起来组成一个完整的工序完工时间表，没有加工的位置为默认的加工该工序的最小值最后一个加工工序的完工时间
    return ret


if __name__ == '__main__':
    # dur = np.array([[1, 2], [3, 4]])
    dur = np.random.randint(1, 10, (3, 3))
    temp1 = np.zeros((3, 3))

    temp1[0, 0] = 1
    temp1[1, 0] = 3
    temp1[1, 1] = 5

    ret = calEndTimeLB(temp1, dur, dur)
