from Params import configs
import numpy as np
from epsGreedyForMch import PredictMch

import torch


def permissibleLeftShift(a_ope, a_mch, a_agv, durMat, mchMat, ST_mch, opIDsOnMchs, ET_mch, unload_time, load_time, CT_agv_pre):
    """

    :param a_ope:
    :param a_mch:
    :param a_agv:
    :param durMat:
    :param mchMat:
    :param ST_mch:
    :param opIDsOnMchs:
    :param ET_mch:
    :param unload_time:
    :param load_time:
    :param CT_agv_pre:
    :return:
    """
    rdyTime_job, rdyTime_mch, ET_unload, ST_load = calJobAndMchRdyTimeOfa(a_ope, a_mch, a_agv, mchMat, durMat, ST_mch, opIDsOnMchs,
                                                      unload_time, load_time, CT_agv_pre)
    if load_time != 0:
        ST_unload = ET_unload - unload_time
        ET_load = rdyTime_job
        CT_agv_updated = rdyTime_job
    else:
        CT_agv_updated = CT_agv_pre
        ST_unload = CT_agv_pre
        ET_unload = CT_agv_pre
        ST_load = CT_agv_pre
        ET_load = CT_agv_pre
    pt_ope = durMat[a_ope // durMat.shape[1]][a_ope % durMat.shape[1]][a_mch]
    startTimesForMchOfa = ST_mch[a_mch]  # 机器a_mch的start数组
    endTimesForMchOfa = ET_mch[a_mch]
    opsIDsForMchOfa = opIDsOnMchs[a_mch]  # 机器a_mch加工工序的编号
    flag = False
    possiblePos = np.where(rdyTime_job < startTimesForMchOfa)[0]
    # machine中以工序的开始时间大于job中action的上一个工序的完工时间
    if len(possiblePos) == 0:
        ST_ope = putInTheEnd(a_ope, rdyTime_job, rdyTime_mch, startTimesForMchOfa, opsIDsForMchOfa, endTimesForMchOfa, pt_ope)
    else:
        idxLegalPos, legalPos, endTimesForPossiblePos = calLegalPos(pt_ope, a_mch, rdyTime_job, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa)
        if len(legalPos) == 0:  # 机器中这个可能位置加工不了该工序，将该工序放到机器的最后一道工序去
            ST_ope = putInTheEnd(a_ope, rdyTime_job, rdyTime_mch, startTimesForMchOfa, opsIDsForMchOfa, endTimesForMchOfa, pt_ope)
        else:  # 该工序在可能的那个位置执行插入操作
            flag = True
            ST_ope = putInBetween(a_ope, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa, endTimesForMchOfa, pt_ope)
    return ST_ope, CT_agv_updated, flag, [ST_unload, ET_unload, ST_load, ET_load]


def putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa, endtineformch0fa, dur_a):
    # index = first position of -config.high in startTimesForMchOfa
    index = np.where(startTimesForMchOfa == -configs.high)[0][0]  # 找到最早能开始加工的位置
    startTime_a = max(jobRdyTime_a, mchRdyTime_a)  # 工序最早的开工时间为max(工件就绪时间, 机器就绪时间)
    startTimesForMchOfa[index] = startTime_a  # 机器对应位置的开始时间置为startTime_a
    opsIDsForMchOfa[index] = a
    endtineformch0fa[index] = startTime_a + dur_a
    return startTime_a


def calLegalPos(dur_a, mch_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa):
    startTimesOfPossiblePos = startTimesForMchOfa[possiblePos]  # possiblepos有可能是一个有可能是多个task，找到machine中tasks的starttimefomach
    durOfPossiblePos = []
    for possiblePo in possiblePos:
        durOfPossiblePos.append(durMat[opsIDsForMchOfa[possiblePo]//durMat.shape[1]][opsIDsForMchOfa[possiblePo] % durMat.shape[1]][mch_a])
    durOfPossiblePos = np.array(durOfPossiblePos)  # 机器上这个位置的工序的加工时间
    # 可能的最早开始加工该工序的时间=max(工序就绪时间, 机器上这个位置的前一个工序加工的结束时间)
    startTimeEarliest = max(jobRdyTime_a, startTimesForMchOfa[possiblePos[0] - 1] +
                            durMat[opsIDsForMchOfa[possiblePos[0]-1]//durMat.shape[1]][opsIDsForMchOfa[possiblePos[0]-1] % durMat.shape[1]][mch_a])
    endTimesForPossiblePos = np.append(startTimeEarliest, (startTimesOfPossiblePos + durOfPossiblePos))[:-1]# end time for last ops don't care
    possibleGaps = startTimesOfPossiblePos - endTimesForPossiblePos
    idxLegalPos = np.where(dur_a <= possibleGaps)[0]
    legalPos = np.take(possiblePos, idxLegalPos)
    return idxLegalPos, legalPos, endTimesForPossiblePos


def putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa,endtineformch0fa,dur_a):
    earlstIdx = idxLegalPos[0]
    # print('idxLegalPos:', idxLegalPos)
    earlstPos = legalPos[0]
    startTime_a = endTimesForPossiblePos[earlstIdx]
    # print('endTimesForPossiblePos:', endTimesForPossiblePos)
    startTimesForMchOfa[:] = np.insert(startTimesForMchOfa, earlstPos, startTime_a)[:-1]
    endtineformch0fa[:]=np.insert(endtineformch0fa, earlstPos, startTime_a+dur_a)[:-1]
    opsIDsForMchOfa[:] = np.insert(opsIDsForMchOfa, earlstPos, a)[:-1]

    return startTime_a


def calJobAndMchRdyTimeOfa(a_ope, a_mch, a_agv, mchMat, durMat, mchsStartTimes, opIDsOnMchs, unload_time, load_time, CT_agv_pre):
    """
    返回工件就绪时间，机器就绪时间
    机器就绪时间为上一个在这个机器上加工的工序的结束时间
    工件就绪时间要根据工件前一个工序的结束时间和AGV将工件运输到该机器的时间进行计算
    :param a_ope:
    :param a_mch:
    :param a_agv:
    :param mchMat:
    :param durMat:
    :param mchsStartTimes:
    :param opIDsOnMchs:
    :param unload_time:
    :param load_time:
    :param CT_agv_pre:
    :return:
    """
    # numpy.take（a，indices，axis = None，out = None，mode ='raise' ）取矩阵中所有元素的第a个元素
    # a % mchMat.shape[1] = 0即该job调度完成或为第一个调度的工序，此时前一个工序为None
    pre_ope = a_ope - 1 if a_ope % durMat.shape[1] != 0 else None  # 工件的前一个工序
    # AGV就绪时间 = AGV的结束时间 + AGV到工件上一个工序加工机器位置空载所用的时间
    rdyTime_agv = CT_agv_pre + unload_time
    # 计算 jobRdyTime_job
    if pre_ope is not None:
        mch_preOpe = np.take(mchMat, pre_ope)  # 根据pre_ope找加工该工件前一个工序的机器
        pt_preOpe = durMat[pre_ope // durMat.shape[1], pre_ope % durMat.shape[1], mch_preOpe]  # 加工时间
        ET_preOpe = (mchsStartTimes[mch_preOpe][np.where(opIDsOnMchs[mch_preOpe] == pre_ope)] + pt_preOpe).item()  # opIDsOnMchs->对应mchJobPredecessor----shape（machine,n_job）
        # 找到数组opIDsOnMchs[mchJobPredecessor]中等于jobPredecessor的索引值####opIDsOnMchs->shape(machine,job)
        ST_load = max(ET_preOpe, rdyTime_agv)  # AGV负载开始的时间=max(AGV就绪时间, 工件上一个工序加工结束时间)
    else:
        ST_load = rdyTime_agv
    ET_unload = ST_load
    ET_load = ST_load + load_time  # AGV负载结束时间 = ST_load + 将工件从前一个工序所在位置负载运输到当前加工位置花费的时间
    rdyTime_job = ET_load
    # 机器加工的前一个工序
    preOpe_mch = opIDsOnMchs[a_mch][np.where(opIDsOnMchs[a_mch] >= 0)][-1] if len(np.where(opIDsOnMchs[a_mch] >= 0)[0]) != 0 else None
    if preOpe_mch is not None:
        durMch_preOpe = durMat[preOpe_mch // durMat.shape[1], preOpe_mch % durMat.shape[1], a_mch]  # 机器加工的前一个工序的时间
        #print('mchfortasktime',ST_mchs[mch_a][np.where(ST_mchs[mch_a] >= 0)][-1] + durMchPredecessor,durMchPredecessor)
        rdyTime_mch = (mchsStartTimes[a_mch][np.where(mchsStartTimes[a_mch] >= 0)][-1] + durMch_preOpe).item()
        #np.where()返回一个索引数组，这里返回在该machine中以调度task的索引。最后返回machine中action上一个task的结束时间
    else:
        rdyTime_mch = 0
    return rdyTime_job, rdyTime_mch, ET_unload, ST_load



if __name__ == "__main__":
    from FJSPT_Env import FJSPT
    from uniform_instance import uni_instance_gen, FJSPDataset
    import time
    from torch.utils.data import DataLoader
    n_j = 3
    n_m = 3
    n_a = 3
    low = -99
    high = 99
    SEED = 200
    #np.random.seed(SEED)
    t3 = time.time()
    train_dataset = FJSPDataset(n_j, n_m, low, high, 2)

    data_loader = DataLoader(train_dataset, batch_size=2)
    for batch_idx, data_set in enumerate(data_loader):
        data_set = data_set.numpy()
        #print(data_set[0])

        #print(t4)
        batch_size = data_set.shape[0]

        env = FJSPT(n_j=n_j, n_m=n_m, n_a=n_a, n_o=n_m)

         # rollout env random action
        t1 = time.time()
        #data = uni_instance_gen(n_j=n_j, n_m=n_m, low=low, high=high,seed=SEED)


        # start time of operations on machines
        mchsStartTimes = -configs.high * np.ones((n_m,n_m*n_j), dtype=np.int32)
        mchsEndtTimes = -configs.high * np.ones((n_m, n_m * n_j), dtype=np.int32)
        # Ops ID on machines
        opIDsOnMchs = -n_j * np.ones([n_m,n_m*n_j], dtype=np.int32)

        # random rollout to test
        # count = 0
        adj, _, omega, mask,mch_mask,_,mch_time,_ = env.reset(data_set)
        print(adj)
        print(data_set)
        #print(env.adj)
        mch_mask = mch_mask.reshape(batch_size, -1,n_m)
        job = omega
        rewards = []
        flags = []
        # ts = []
        #print(env.mask_mch[0])
        while True:
            action = []
            mch_a = []
            for i in range(batch_size):

                a= np.random.choice(omega[i][np.where(mask[i] == 0)])


                #index = np.where(job[i] == a)[0].item()


                m = np.random.choice(np.where(mch_mask[i][a] == 0)[0])

                action.append(a)
                mch_a.append(m)

            '''mch_a = np.random.choice()
            mch_a = PredictMch(env,action,1)'''

            '''row = action // n_j  # 取整除
            col = action % n_m  # 取余数
            job_time=data_set[0][row][col]

            mch_a=np.random.choice(np.where(job_time>0)[0])'''


            #dur_a=data[row][col][mch_a]

            # print(mch_a)
            # print('action:', action)
            # t3 = time.time()
            #print('env_opIDOnMchs\n', env.opIDsOnMchs)
            #print('11',env.mchsEndTimes[0])
            adj, _, reward, done, omega, mask,job,_,mch_time,_= env.step(action, mch_a)

            #print('33',env.mchsEndTimes[0])
            #print('reward',reward[0],env.dur_a)
            # t4 = time.time()
            # ts.append(t4 - t3)
            #jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(a=action,mch_a=mch_a, mchMat=m, durMat=data, ST_mchs=ST_mchs, opIDsOnMchs=opIDsOnMchs)
            #print('mchRdyTime_a:', mchRdyTime_a,"\n",'jobrdytime',jobRdyTime_a)

            #startTime_a, flag = permissibleLeftShift(a=action, mch_a=mch_a,durMat=data.astype(np.single), mchMat=m, ST_mchs=ST_mchs, opIDsOnMchs=opIDsOnMchs,mchEndTime=mchsEndtTimes,dur_a=dur_a)
            #flags.append(flag)

            # print('startTime_a:', startTime_a)
            # print('ST_mchs\n', ST_mchs)
            # print('NOOOOOOOOOOOOO' if not np.array_equal(env.ST_mchs, ST_mchs) else '\n')
            #print('opIDsOnMchs\n', opIDsOnMchs)

            # print('LBs\n', env.LBs)
            rewards.append(reward)
            # print('ET after action:\n', env.LBs)
            #print()
            if env.done():
                break
        t2 = time.time()
        print(t2 - t1)
        # print(sum(ts))
        # print(np.sum(opIDsOnMchs // n_m, axis=1))
        # print(np.where(ST_mchs == ST_mchs.max()))
        # print(opIDsOnMchs[np.where(ST_mchs == ST_mchs.max())])
        #print(ST_mchs.max() + np.take(data[0], opIDsOnMchs[np.where(ST_mchs == ST_mchs.max())]))
        # np.save('sol', opIDsOnMchs // n_m)
        # np.save('jobSequence', opIDsOnMchs)
        # np.save('testData', data)
        # print(ST_mchs)

        #print(data)

        print()

        print(env.ST_mchs)
        print('reward---------------', env.ET_mchs, env.ET_mchs.max(-1).max(-1))
        print()
        print(env.opesOnMchs[0])
        print(env.adj[0])
        # print(sum(flags))
        # data = np.load('data.npy')
        t4 = time.time() - t3
        print(t4)
        # print(len(np.where(np.array(rewards) == 0)[0]))
        # print(rewards)
