import math

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import os
from torch.utils.data import DataLoader
from torch.nn import DataParallel


def permute_rows(x):
    """
    实现了对输入的 NumPy 数组 x 进行行的随机排列
    """
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, low, high, seed=None):
    if seed != None:
        np.random.seed(seed)

    time0 = np.random.randint(low=low, high=high, size=(n_j, n_m, n_m - 1))
    time1 = np.random.randint(low=1, high=high, size=(n_j, n_m, 1))
    times = np.concatenate((time0, time1), -1)
    for i in range(n_j):
        times[i] = permute_rows(times[i])
    return times


# -99~99 randint -- 5
# -99~99 0~99 uniform -- 0
# -99~99 1~99 uniform -- 0


class FJSPDataset(Dataset):

    def __init__(self, n_j, n_m, low, high, num_samples=1000000, seed=None, offset=0, distribution=None):
        super(FJSPDataset, self).__init__()
        self.data_set = []
        if seed != None:
            np.random.seed(seed)
        time0 = np.random.uniform(low=low, high=high, size=(num_samples, n_j, n_m, n_m - 1))
        for i in range(time0.shape[0]):
            for j in range(time0.shape[1]):
                for k in range(time0.shape[2]):
                    for l in range(time0.shape[3]):
                        if time0[i][j][k][l] <= 5 and time0[i][j][k][l] > 0:
                            time0[i][j][k][l] = -time0[i][j][k][l]
        # 保证每个工件至少能在一个机器上加工
        time1 = np.random.uniform(low=5, high=high, size=(num_samples, n_j, n_m, 1))
        times = np.concatenate((time0, time1), -1)
        for j in range(num_samples):
            for i in range(n_j):
                times[j][i] = permute_rows(times[j][i])
            # Sample points randomly in [0, 1] square
        self.data = np.array(times)
        self.size = len(self.data)

    def getdata(self):
        return self.data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


def override(fn):
    """
    override decorator
    """
    return fn


def get_travel_time(mch_num=30):
    # 设置随机数种子
    np.random.seed(42)
    tt = np.random.uniform(low=5, high=10, size=(mch_num, mch_num))
    # 对随机数保留一位小数
    tt = np.round(tt, decimals=1)
    # 将对角线元素置为0
    np.fill_diagonal(tt, 0)
    return tt



if __name__ == '__main__':
    n_j = 10
    n_m = 8
    n_a = 2
    low = -99
    high = 99
    num_samples = 1
    seed = 200
    dataset = FJSPDataset(n_j=n_j, n_m=n_m, low=low, high=high, num_samples=num_samples, seed=seed, )
    # print()
    pt = []

    for i in range(num_samples):
        for j in range(n_j):
            pt_j = []
            for k in range(5):
                pt_o = []
                idx = []
                for l in range(n_m):
                    dataset.data[i][j][k][l] = math.ceil(dataset.data[i][j][k][l])
                    if dataset.data[i][j][k][l] > 0:
                        idx.append(l)
                        pt_o.append(int(dataset.data[i][j][k][l]))
                pt_j.append([idx, pt_o])
            pt.append(pt_j)


    np.save('Ins{}J{}M{}A{}.npy'.format(num_samples, n_j, n_m, n_a), np.array(dataset))
    # data1为list类型，参数index为索引，column为列名
    # pt_row = []
    # for j in pt:
    #     for r in j:
    #         pt_row.append(r)
    # data2 = pd.DataFrame(data=pt_row, index=None)
    # # PATH为导出文件的路径和文件名
    # data2.to_csv("PATH.csv")
    #
    #
    # tt = get_travel_time()
    # np.save('travel_time.npy', tt)
