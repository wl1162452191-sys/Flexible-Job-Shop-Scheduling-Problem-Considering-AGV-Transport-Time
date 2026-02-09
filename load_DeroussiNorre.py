"""
@Project ：FJSP_MultiPPO 
@File    ：load_DeroussiNorre.py
@IDE     ：PyCharm 
@Author  ：lyon
@Date    ：27/03/2024 20:44 
@Des     ：
"""
import numpy as np


def load_travel_time(path):
    TT = []
    with open(path) as f:
        data = f.readlines()
    for line in data:
        line = line.strip("\n")  # 去除末尾的换行符
        data_split = line.split(' ')
        temp = list(map(int, data_split))
        TT.append(temp)
    return TT


def load_deroussi_norre(path):
    array = []
    with open(path) as f:
        data = f.readlines()
    for line in data:
        line = line.strip("\n")  # 去除末尾的换行符
        data_split = list(filter(None, line.split(' ')))
        temp = list(map(int, data_split))
        array.append(temp)

    J_num, M_num, AGV_num = array[0][0], array[0][1], array[0][2]
    Op_num = []
    for i in range(1, len(array)):
        Op_num.append(array[i][0])

    PT = []

    for i in range(J_num):
        for j in range(Op_num[i]):
            Job_i = []
            for j in range(Op_num[i]):
                Job_i.append([] * M_num)
            PT.append(Job_i)

    for idx, job in enumerate(array):
        if idx == 0:
            continue
        for Op in range(job[0]):
            machine_num = job[Op * 4 + 1]  # 兼容机器的数量
            machine = job[2 + Op * 4: 2 + Op * 4 + machine_num]#
            PT[idx - 1][Op].append([x for x in machine])
            PT[idx - 1][Op].append(machine_num * [job[4 + Op * 4]])

    Op_dic = dict(enumerate(Op_num))
    O_num = sum(Op_num)
    arrive_time = [0 for i in range(J_num)]
    time_window = np.zeros(shape=(J_num, max(Op_num), M_num))
    for job_id, i in enumerate(PT):
        for ope_id, j in enumerate(i):
            id_m = j[0]
            pt_m = j[1]
            for k in range(len(id_m)):
                time_window[job_id][ope_id][id_m[k] - 1] = pt_m[k]
    # return PT, M_num, Op_dic, O_num, J_num, AGV_num, arrive_time, due_time
    num_ope = [Op_dic[key] for key in Op_dic]
    return time_window, num_ope
def load_standard_instance(path):
    array = []
    with open(path) as f:
        data = f.readlines()
    for line in data:
        line = line.replace("\t", " ")  # 去除所有制表符
        line = line.strip("\n")  # 去除末尾的换行符
        data_split = list(filter(None, line.split(' ')))
        temp = list(map(int, data_split))
        array.append(temp)

    J_num, M_num, AGV_num = array[0][0], array[0][1], array[0][2]
    Op_num = []
    for i in range(1, len(array)):
        Op_num.append(array[i][0])

    PT = []

    for i in range(J_num):
        Job_i = []
        for j in range(Op_num[i]):
            Job_i.append([] * M_num)
        PT.append(Job_i)

    for idx, job in enumerate(array):
        index = 0
        if idx == 0:
            continue
        for Op in range(job[0]):
            machine_num = job[index + 1]  # 兼容机器的数量
            index = index + 1
            machine_set = []
            machine_time = []
            for m in range(machine_num):
                machine = job[index + 1]
                processing_time = job[index + 2]
                index = index + 2
                machine_set.append(machine)
                machine_time.append(processing_time)

            PT[idx - 1][Op].append(machine_set)
            PT[idx - 1][Op].append(machine_time)

    Op_dic = dict(enumerate(Op_num))
    # O_num = sum(Op_num)
    # arrive_time = [0 for i in range(J_num)]
    time_window = np.zeros(shape=(J_num, max(Op_num), M_num))
    for job_id, i in enumerate(PT):
        for ope_id, j in enumerate(i):
            id_m = j[0]
            pt_m = j[1]
            for k in range(len(id_m)):
                time_window[job_id][ope_id][id_m[k] - 1] = pt_m[k]
    # return PT, M_num, Op_dic, O_num, J_num, AGV_num, arrive_time, due_time
    num_ope = [Op_dic[key] for key in Op_dic]
    return time_window, num_ope

if __name__ == '__main__':
    tt = load_travel_time("FJSPTinstances/DeroussiNorre/travel_time.txt")
    tt1 = load_deroussi_norre("FJSPTinstances/DeroussiNorre/fjsp1.txt")
