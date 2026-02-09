import random
from copy import deepcopy

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gym.utils import EzPickle

from Params import configs
from min_job_machine_time import min_job_mch
from permissibleLS import permissibleLeftShift
from uniform_instance import override
from updateAdjMat import getMchNbghs
from updateEndTimeLB import calEndTimeLB, calEndTimeLBm


def get_travel_time(mch_num=30):
    # 设置随机数种子
    np.random.seed(42)
    tt = np.random.uniform(low=5, high=10, size=(mch_num, mch_num))
    # 对随机数保留一位小数
    tt = np.round(tt, decimals=1)
    # 将对角线元素置为0
    np.fill_diagonal(tt, 0)
    np.save('travel_time.npy', tt)
    return tt


class FJSPT(gym.Env, EzPickle):
    def __init__(self, n_j, n_m, n_a, n_o, last_ope_ids=None, travel_time=None):
        EzPickle.__init__(self)

        self.LBm = None
        self.LBs = None
        self.posRewards = None
        self.dur_cp = None
        self.dur = None
        self.batch_size = None
        self.flags = None
        self.initQuality = None
        self.max_endTime = None
        self.step_count = 0
        self.getEndTimeLB = calEndTimeLB
        # 机器相关
        self.num_mch = n_m
        self.getNghbs = getMchNbghs
        self.mch_sequence = None    # 记录加工每个工序的机器
        self.ST_mchs = None
        self.ET_mchs = None  # 这不就是最大完工时间吗
        self.CT_mch = None
        self.up_mchendtime = None
        self.opesOnMchs = None
        # 工件相关
        self.num_jobs = n_j
        self.first_ope_ids = []
        if last_ope_ids != None:
            self.last_ope_ids = last_ope_ids
        else:
            self.last_ope_ids = []
        self.num_total_ope = n_j * n_o
        self.num_ope = n_o
        self.ope_sequence = None  # 工序序列，工序选择的顺序
        self.pt_ope_min = None
        self.pt_ope_mean = None
        self.pt_ope_2d = None
        self.pt_a_ope = None
        self.CT_job = None
        self.CT_ope = None
        self.ope_ids = None
        self.mask_job = None
        self.mask_ope_mch = None
        self.finished_ope = None
        self.loca_jobs = []
        # Agv相关
        self.num_agv = n_a
        if travel_time is None:
            self.TT = get_travel_time()
        else:
            self.TT = travel_time
        self.agv_sequence = None  # 记录运输每个工序的AGV
        self.CT_agv = None
        self.ST_unload = []
        self.ET_unload = []
        self.ST_load = []
        self.ET_load = []
        self.loca_start = []
        self.loca_end = []
        self.loca_agv = []
        self.U_agv = []
        self.opeOnAgv = None
        self.plt_cnt = 0

    def done(self):
        if np.all(self.ope_sequence[0] >= 0):
            return True
        return False

    @override
    def step(self, a_ope, a_mch, a_agv, gantt_plt=None):
        feas, rewards, dones, masks_job, masks_mch = [], [], [], [], []
        mch_spaces, opes_min_ct = [], []
        for i in range(self.batch_size):
            # redundant action makes no effect 重复的工序动作无效
            if a_ope[i] not in self.ope_sequence[i]:
                job_idx = a_ope[i] // self.num_ope  # 取整除，第几个工件
                ope_idx = a_ope[i] % self.num_ope  # 取余数，工件的第几道工序
                agv_idx = a_agv[i].item()
                if i == 0:
                    self.step_count += 1  # 执行step的次数
                self.finished_ope[i, job_idx, ope_idx] = 1  # 工件job_idx的第ope_idx道工序加工完成，对应位置置1
                self.pt_a_ope = self.dur[i, job_idx, ope_idx, a_mch[i]]
                # 找到self.partial_sol_sequence[i]中第一次出现小于0的元素的位置索引，将其设置为a_ope[i]
                self.ope_sequence[i][np.where(self.ope_sequence[i] < 0)[0][0]] = a_ope[i]
                self.mch_sequence[i][job_idx][ope_idx] = a_mch[i]  # 选中的机器的编号
                self.agv_sequence[i][job_idx][ope_idx] = agv_idx  # 选中的机器的编号

                loca_agv = self.loca_agv[i][agv_idx][-1]
                loca_mch_pre = 0 if ope_idx == 0 else self.mch_sequence[i][job_idx][ope_idx - 1] + 1  # 0表示在LU
                mch_cur = a_mch[i].item()
                loca_mch_cur = mch_cur + 1  # 序号加一表示这个机器所在的位置编号
                # print("M{}".format(loca_agv), "M{}".format(loca_mch_pre), "M{}".format(loca_mch_cur))
                if loca_mch_cur != loca_mch_pre:  # 没有使用AGV
                    self.loca_agv[i][agv_idx].append(loca_mch_pre)
                    self.loca_agv[i][agv_idx].append(loca_mch_cur)
                unload_time = self.TT[loca_agv, loca_mch_pre]
                load_time = self.TT[loca_mch_pre, loca_mch_cur]
                CT_agv_pre = self.CT_agv[i][agv_idx]
                # UPDATE STATE:
                # permissible left shift 允许向左移动
                ST_ope, CT_agv, flag, unload_load_time = permissibleLeftShift(a_ope[i], a_mch=a_mch[i], a_agv=agv_idx,
                                                            durMat=self.dur_cp[i], mchMat=self.mch_sequence[i],
                                                            ST_mch=self.ST_mchs[i], opIDsOnMchs=self.opesOnMchs[i],
                                                            ET_mch=self.ET_mchs[i], unload_time=unload_time,
                                                            load_time=load_time, CT_agv_pre=CT_agv_pre)
                self.flags.append(flag)
                if gantt_plt is not None and i == self.batch_size - 1:  # 将每次批处理中第一个的甘特图输出来
                    gantt_plt.gantt_plt_ope(job_idx, ope_idx, mch_cur, agv_idx, ST_ope, self.pt_a_ope, self.num_jobs,
                                            self.num_mch, unload_load_time)
                # update omega or mask
                if a_ope[i] not in self.last_ope_ids[i]:  # 该工序不是对应工件最后一个工序
                    self.ope_ids[i, job_idx] += 1
                else:  # 该工件的所有工序都已经加工完毕，工件蒙版对应的位置置1，表示以后不能再选择该工件了
                    self.mask_job[i, job_idx] = 1

                self.CT_ope[i, job_idx, ope_idx] = ST_ope + self.pt_a_ope  # 该工件对应工序的最大完工时间
                self.CT_agv[i, agv_idx] = CT_agv  # 该AGV的最大运输时间
                self.ST_unload[i][agv_idx].append(unload_load_time[0])
                self.ET_unload[i][agv_idx].append(unload_load_time[1])
                self.ST_load[i][agv_idx].append(unload_load_time[2])
                self.ET_load[i][agv_idx].append(unload_load_time[3])
                self.LBs[i] = calEndTimeLB(self.CT_ope[i], self.pt_ope_min[i], self.pt_ope_mean[i])
                self.LBm[i] = calEndTimeLBm(self.CT_ope[i], self.pt_ope_min[i])
                self.U_agv[i][agv_idx] = (sum(self.ET_unload[i][agv_idx]) + sum(self.ET_load[i][agv_idx]) -
                                          sum(self.ST_unload[i][agv_idx]) - sum(self.ST_load[i][agv_idx])) / self.LBm[i].max()
                # adj matrix 邻接矩阵，析取图添加弧策略
                pre_ope_mch, succd_ope_mch = self.getNghbs(a_ope[i], self.opesOnMchs[i])
                # self.adj[i, a_ope[i]] = 0 fixme 这下面几步好像是重复的，是不是注释掉也一样呢？？
                # self.adj[i, a_ope[i], a_ope[i]] = 1
                # if a_ope[i] not in self.first_ope_ids[i]:  # 该工序不是第一个工序，将邻接矩阵对应的位置置1
                #     self.adj[i, a_ope[i], a_ope[i] - 1] = 1
                self.adj[i, pre_ope_mch, succd_ope_mch] = 0  # 相当于在中间插了一个工序，要把之前这两个析取弧断开
                self.adj[i, a_ope[i], pre_ope_mch] = 1
                self.adj[i, succd_ope_mch, a_ope[i]] = 1
                done = self.done()
                mch_space, ope_min_ct, mask_job1, mch_mask = min_job_mch(self.CT_mch[i], self.CT_job[i],
                                                                         self.ET_mchs[i], self.num_mch,
                                                                         self.dur_cp[i], self.CT_ope[i],
                                                                         self.ope_ids[i], self.mask_job[i], done,
                                                                         self.mask_ope_mch[i])
                mch_spaces.append(mch_space)
                opes_min_ct.append(ope_min_ct)
                masks_job.append(mask_job1)
                masks_mch.append(mch_mask)
            fea = np.concatenate((self.LBm[i].reshape(-1, 1) / configs.et_normalize_coef,
                                  self.finished_ope[i].reshape(-1, 1)), axis=-1)  # [工序完工时间表,工序完工标志] 组成特征矩阵
            feas.append(fea)
            reward = -(self.LBm[i].max() - self.max_endTime[i])  # -(预计完工时间的最大值 - 前一步预计完工时间的最大值)
            if reward == 0:
                reward = configs.rewardscale  # 让奖励不是从0开始？？
                self.posRewards[i] += reward
            rewards.append(reward)
            self.max_endTime[i] = self.LBm[i].max()
            dones.append(done)
        return self.adj, np.array(feas), rewards, dones, self.ope_ids, masks_job, opes_min_ct, self.mask_ope_mch, self.CT_mch, self.CT_job

    @override
    def reset(self, data):
        self.batch_size = data.shape[0]
        flag_last_ope = False
        if len(self.last_ope_ids) == 0:
            flag_last_ope = True
        for i in range(self.batch_size):
            first_ope_id = np.arange(start=0, stop=self.num_total_ope, step=1)\
                               .reshape(self.num_jobs, -1)[:, 0]
            self.first_ope_ids.append(first_ope_id)
            last_ope_id = np.arange(start=0, stop=self.num_total_ope, step=1) \
                              .reshape(self.num_jobs, -1)[:, -1]
            if flag_last_ope:
                self.last_ope_ids.append(last_ope_id)
        self.first_ope_ids = np.array(self.first_ope_ids)
        self.last_ope_ids = np.array(self.last_ope_ids)
        self.step_count = 0
        self.dur = data.astype(np.single) # single单精度浮点数
        self.dur_cp = deepcopy(self.dur)
        # 记录每一步选择的工序编号，动作空间大小为n_j*n_o
        self.ope_sequence = -1 * np.ones((self.batch_size, sum([self.last_ope_ids[0][i] - self.first_ope_ids[0][i] + 1
                                                                for i in range(self.num_jobs)])), dtype=int)
        # 记录每个工件工序每次选择的机器
        self.mch_sequence = -1 * np.ones((self.batch_size, self.num_jobs, self.num_ope), dtype=int)
        # 记录每一步选择的AGV
        self.agv_sequence = -1 * np.ones((self.batch_size, self.num_jobs, self.num_ope), dtype=int)
        self.flags = []
        self.posRewards = np.zeros(self.batch_size)
        self.adj = []
        # initialize adj matrix, 图的邻接矩阵, 大小为 工序数*工序数，有向图
        for i in range(self.batch_size):
            # 创建一个number_of_operations*number_of_operations的二维数组，对角线以下的对角线上的元素为1，其他元素为0，k是对角线偏移量
            self_as_nei = np.eye(self.num_total_ope, dtype=np.single)
            conj_nei_up_stream = np.eye(self.num_total_ope, k=-1, dtype=np.single)  # 行工序在列工序后面为1，比如第1个工序在第0个工序后，那么邻接矩阵的第1行第0列的值为1
            # first column does not have upper stream conj_nei，每个工件的第一个工序没有前工序，因此对应那行的值置为0
            conj_nei_up_stream[self.first_ope_ids] = 0
            adj = self_as_nei + conj_nei_up_stream
            self.adj.append(adj)
        self.adj = torch.tensor(np.array(self.adj))
        # initialize features
        self.mask_ope_mch = np.full(
            shape=(self.batch_size, self.num_jobs, self.num_ope, self.num_mch),
            fill_value=0,
            dtype=bool)
        pt_ope_min = []
        pt_ope_mean = []
        for t in range(self.batch_size):
            min = []
            mean = []
            for i in range(self.num_jobs):
                dur_min = []
                dur_mean = []
                for j in range(self.num_ope):
                    durmch = self.dur[t][i][j][np.where(self.dur[t][i][j] > 0)]  # 大于0的加工时间
                    self.mask_ope_mch[t][i][j] = [1 if d <= 0 else 0 for d in self.dur_cp[t][i][j]]  # 小于零mask为True，遮住
                    self.dur[t][i][j] = [durmch.mean() if d <= 0 else d for d in
                                         self.dur[t][i][j]]  # 工件不能在该机器上加工，用平均加工时间代替
                    dur_min.append(0 if durmch.size == 0 else durmch.min().tolist())  # 0表示没有这道工序
                    dur_mean.append(0 if durmch.size == 0 else durmch.mean().tolist())  # 0表示没有这道工序
                    # dur_mean.append(durmch.mean().tolist())
                min.append(dur_min)
                mean.append(dur_mean)
            pt_ope_min.append(min)  # 每个工序的最小加工时间
            pt_ope_mean.append(mean)  # 每个工序的平均加工时间
        self.pt_ope_min = np.array(pt_ope_min)  # 未加加工的工序为加工该工序的时间最小值，如果工序加工完成，那么对应位置置0
        self.pt_ope_mean = np.array(pt_ope_mean)
        self.pt_ope_2d = np.concatenate(
            [self.pt_ope_min.reshape((self.batch_size, self.num_jobs, self.num_ope, 1)),
             self.pt_ope_mean.reshape((self.batch_size, self.num_jobs, self.num_ope, 1))], -1)
        self.LBs = np.cumsum(self.pt_ope_2d, -2)  # 对输入的input_2d数组沿着倒数第二个轴进行累积求和，维度保持不变
        # 工序完工时间表：未加工工序的完工时间 = 该工序前一个工序的完工时间 + 这个工序的最小加工时间
        # 已加工工序 = 该工序的最大完工时间
        self.LBm = np.cumsum(self.pt_ope_min, -1)
        self.initQuality = np.ones(self.batch_size)
        for i in range(self.batch_size):
            self.initQuality[i] = self.LBm[i].max() if not configs.init_quality_flag else 0
        self.max_endTime = self.initQuality
        self.CT_job = np.zeros((self.batch_size, self.num_jobs))
        self.finished_ope = np.zeros((self.batch_size, self.num_jobs, self.num_ope), dtype=int)
        fea = np.concatenate((self.LBm.reshape(self.batch_size, -1, 1) / configs.et_normalize_coef,
                              self.finished_ope.reshape(self.batch_size, -1, 1)), axis=-1)
        # initialize feasible omega，工件加工到哪个工序
        self.ope_ids = self.first_ope_ids.astype(int)
        # initialize mask
        self.mask_job = np.full(shape=(self.batch_size, self.num_jobs), fill_value=0, dtype=bool)
        self.CT_mch = np.zeros((self.batch_size, self.num_mch))
        self.CT_agv = np.zeros((self.batch_size, self.num_agv))
        # start time of operations on machines 每个工序的开始加工时间
        self.ST_mchs = -configs.high * np.ones(
            (self.batch_size, self.num_mch, self.num_total_ope))
        self.ET_mchs = -configs.high * np.ones(
            (self.batch_size, self.num_mch, self.num_total_ope))
        self.opesOnMchs = -self.num_jobs * np.ones(
            (self.batch_size, self.num_mch, self.num_total_ope), dtype=int)
        self.up_mchendtime = np.zeros_like(self.ET_mchs)
        self.CT_ope = np.zeros((self.batch_size, self.num_jobs, self.num_ope))
        dur = self.dur.reshape(self.batch_size, -1, self.num_mch)
        self.mask_ope_mch = self.mask_ope_mch.reshape(self.batch_size, -1, self.mask_ope_mch.shape[-1])
        self.U_agv = np.zeros((self.batch_size, self.num_agv))
        self.opeOnAgv = -self.num_jobs * np.ones(
            (self.batch_size, self.num_agv, self.num_total_ope), dtype=int)
        self.loca_agv = [[[0] for _ in range(self.num_agv)] for _ in range(self.batch_size)]  # 0表示在LU
        self.loca_jobs = [[[0] for _ in range(self.num_jobs)] for _ in range(self.batch_size)]  # 0表示在LU
        self.ST_unload = [[[] for _ in range(self.num_agv)] for _ in range(self.batch_size)]
        self.ET_unload = [[[] for _ in range(self.num_agv)] for _ in range(self.batch_size)]
        self.ST_load = [[[] for _ in range(self.num_agv)] for _ in range(self.batch_size)]
        self.ET_load = [[[] for _ in range(self.num_agv)] for _ in range(self.batch_size)]
        return self.adj, fea, self.ope_ids, self.mask_job, self.mask_ope_mch, dur, self.CT_mch, self.CT_job, \
            self.CT_agv, self.U_agv


class FJSPTGanttChart():
    def __init__(self, num_job, num_mch, num_agv):
        super(FJSPTGanttChart, self).__init__()

        self.num_job = num_job
        self.num_mch = num_mch
        self.num_agv = num_agv
        self.initialize_plt()

    def colour_gen(self, n):
        """
        为工件生成随机颜色
        :param n: 工件数
        :return: 颜色列表
        """
        color_bits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
        colours = []
        random.seed(234)
        for i in range(n):
            colour_bits = ['#']
            colour_bits.extend(random.sample(color_bits, 6))
            colours.append(''.join(colour_bits))
        return colours

    def initialize_plt(self):
        # plt.figure(figsize=((self.num_job * 1.5 / 1.5 , (self.num_mch + self.num_agv) / 2/ 1.5)))
        plt.figure(figsize=((self.num_job * 1.5 / 2 / 1.2, (self.num_mch + self.num_agv) / 2)))
        y_value = list(range(1, self.num_mch + self.num_agv + 1))
        Job_text = ['J' + str(i + 1) for i in range(self.num_job)]
        AGV_text = ['A' + str(i + 1) for i in range(self.num_agv)]
        Machine_text = ['M' + str(i + 1) for i in range(self.num_mch)]
        y_ticks = ["0"] + Machine_text[: self.num_mch] + AGV_text[: self.num_agv]
        y = range(len(y_ticks))
        plt.yticks(y, y_ticks)
        font_size = 24

        # plt.xlabel('Makespan', size=12, fontdict={'family': 'SimSun'})
        # plt.ylabel('机器号', size=12, fontdict={'family': 'SimSun'})
        plt.yticks(y_value, fontproperties='Times New Roman', size=font_size)
        plt.xticks(fontproperties='Times New Roman', size=font_size)


        # 使用已经存在的颜色创建图例标记
        makespan = 100
        colors = self.colour_gen(self.num_job)
        plt.vlines([makespan], 0, self.num_mch + self.num_agv + 1, linestyles='dashed', colors='red')
        # xy范围
        plt.ylim(0, self.num_mch + self.num_agv + 1)
        plt.text(makespan + 8,
                 0.3,
                 str(makespan),
                 fontsize=font_size,
                 color="r",
                 verticalalignment="top",
                 horizontalalignment="right",
                 fontdict={'family': 'Times New Roman'}
                 )

        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[j]) for j in range(self.num_job)]

        # 添加图例，并指定图例标记，bbox_to_anchor=(0.5, 1.2)原点在左下角，(0.5, 0.5)为中间
        # plt.legend(handles, Job_text, loc='upper center', bbox_to_anchor=(0.5, 1.75), ncol=5)
        plt.legend(handles, Job_text, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=17, prop={'size': 15,
                                                                                                    'family': 'Times New Roman',
                                                                                                    'style': 'normal',})


    def gantt_plt_ope(self, job, operation, machine, agv, ST_ope, dur_ope, num_job, num_mch, unload_load_time):
        """
        绘制甘特图
        :param job: 工件号
        :param operation: 工序号
        :param machine: 机器号
        :param agv: agv号
        :param ST_ope: 开始时间
        :param dur_ope: 加工时间
        :param colors: 颜色列表
        :param num_mch:
        :param num_agv:
        """
        ST_unload, ET_unload, ST_load, ET_load = unload_load_time
        # print(agv, "J{}".format(int(job) + 1), "空载起点", "负载起点", machine, ST_unload, ET_unload - ST_unload, ET_load - ST_load)
        colors = self.colour_gen(num_job)
        plt.barh(y=machine + 1, width=dur_ope, height=0.5, left=ST_ope, color=colors[job], edgecolor='black', linewidth=0.5)  # 绘制工序甘特图
        if ET_unload - ST_unload != 0:
            plt.barh(y=agv + num_mch + 1, width=ET_unload - ST_unload, height=0.5, left=ST_unload, color="white", edgecolor='black', linewidth=0.5)  # 空载AGV甘特图
        if ET_load - ST_load != 0:
            plt.barh(y=agv + num_mch + 1, width=ET_load - ST_load, height=0.5, left=ST_load, color=colors[job], edgecolor='black', linewidth=0.5)  # 负载AGV甘特图
        # plt.text(x=ST_ope + dur_ope / 10, y=machine + 0.9, s='J%s' % (job + 1), size=8)
