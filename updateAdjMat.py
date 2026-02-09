from Params import configs
import numpy as np


def getMchNbghs(ope_a, opIDsOnMchs):
    coordAction = np.where(opIDsOnMchs == ope_a)  # action 位于矩阵中的位置
    # 位于该machine中的前一个工序（除action为第一个工序时除外）
    pre_ope_mch = opIDsOnMchs[coordAction[0], coordAction[1] - 1 if coordAction[1].item() > 0 else coordAction[1]].item()
    # 后面一个工序，如果该工序是最后一个的话就是该工序
    succeOpeTemp = opIDsOnMchs[coordAction[0], coordAction[1] + 1 if coordAction[1].item() + 1 < opIDsOnMchs.shape[-1] else coordAction[1]].item()
    succd_ope_mch = ope_a if succeOpeTemp < 0 else succeOpeTemp  # 位于该machine中的后一个工序（除action为第一个task和下一个task为负外）
    return pre_ope_mch, succd_ope_mch


if __name__ == '__main__':
    opIDsOnMchs = np.array([[7, 29, 33, 16, -6, -6],  # machine1
                            [6, 18, 28, 34, 2, -6],  # machine2
                            [26, 31, 14, 21, 11, 1],
                            [30, 19, 27, 13, 10, -6],
                            [25, 20, 9, 15, -6, -6],
                            [24, 12, 8, 32, 0, -6]])
    print(opIDsOnMchs.shape[-1])

    action = 29
    precd, succd = getMchNbghs(action, opIDsOnMchs)
    print(precd, succd)
    print(opIDsOnMchs)
