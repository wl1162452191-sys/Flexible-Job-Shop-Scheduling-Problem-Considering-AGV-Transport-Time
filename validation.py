import pandas as pd

from DataRead import getdata
from epsGreedyForMch import PredictMch
from load_DeroussiNorre import load_travel_time, load_deroussi_norre, load_standard_instance
from mb_agg import *
from Params import configs
from copy import deepcopy
from FJSPT_Env import FJSPT, FJSPTGanttChart
from mb_agg import g_pool_cal
import copy
from agent_utils import sample_select_action
from agent_utils import greedy_select_action
import numpy as np
import torch
import matplotlib.pyplot as plt
from Params import configs


def validate(vali_set, batch_size, policy_jo, policy_mc, policy_agv, n_j=configs.n_j, n_m=configs.n_m, n_a=configs.n_a,
             n_o=configs.n_m, last_ope_ids=None, travel_time1=None):
    policy_job = copy.deepcopy(policy_jo)
    policy_mch = copy.deepcopy(policy_mc)
    policy_agv = copy.deepcopy(policy_agv)
    policy_job.eval()
    policy_mch.eval()
    policy_agv.eval()

    def eval_model_bat(bat, i, plt_gantt=False):
        C_max = []
        with torch.no_grad():
            data = bat.numpy()
            env = FJSPT(n_j=n_j, n_m=n_m, n_a=n_a, n_o=n_o, last_ope_ids=last_ope_ids, travel_time=travel_time1)
            gantt_chart = FJSPTGanttChart(n_j, n_m, n_a) if plt_gantt else None
            device = torch.device(configs.device)
            g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                                     batch_size=torch.Size(
                                         [batch_size, n_j * n_o, n_j * n_o]),
                                     n_nodes=n_j * n_o,
                                     device=device)
            adj, fea, ope_ids, mask_job, mask_ope_mch, dur, CT_mch, CT_job, CT_agv, U_agv = env.reset(data)
            j = 0
            ep_rewards = - env.initQuality
            rewards = []
            env_mask_mch = torch.from_numpy(np.copy(mask_ope_mch)).to(device)
            env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
            mch_pooled = None
            agv_pooled = None

            while True:
                env_adj = aggr_obs(deepcopy(adj).to(device).to_sparse(), n_j * n_o)
                env_fea = torch.from_numpy(np.copy(fea)).float().to(device)
                env_fea = deepcopy(env_fea).reshape(-1, env_fea.size(-1))
                env_candidate = torch.from_numpy(np.copy(ope_ids)).long().to(device)
                env_mask_job = torch.from_numpy(np.copy(mask_job)).to(device)
                env_CT_mch = torch.from_numpy(np.copy(CT_mch)).float().to(device)
                env_CT_agv = torch.from_numpy(np.copy(CT_agv)).float().to(device)
                env_U_agv = torch.from_numpy(np.copy(U_agv)).float().to(device)
                # env_job_time = torch.from_numpy(np.copy(job_time)).float().to(device)
                a_ope, a_idx, log_a, action_node, _, mask_mch_action, hx = policy_job(x=env_fea,
                                                                                      graph_pool=g_pool_step,
                                                                                      padded_nei=None,
                                                                                      adj=env_adj,
                                                                                      ope_ids=env_candidate,
                                                                                      mask_job=env_mask_job,
                                                                                      mask_ope_mch=env_mask_mch,
                                                                                      dur=env_dur,
                                                                                      a_index=0,
                                                                                      old_action=0,
                                                                                      mch_pooled=mch_pooled,
                                                                                      agv_pooled=agv_pooled,
                                                                                      old_policy=True,
                                                                                      T=1,
                                                                                      greedy=True
                                                                                      )
                # if CT_job[0][a_ope // 5] > 0 and CT_job[0][a_ope // 5] < 300:
                #     mask_mch_action[0][0][3] = torch.tensor(True, device='cuda')
                pi_mch, mch_pooled, _, _ = policy_mch(pt_ope=action_node,
                                                      hx_ope=hx,
                                                      agv_pooled=agv_pooled,
                                                      mask_mch_action=mask_mch_action,
                                                      CT_mch=env_CT_mch,
                                                      greedy=True)
                _, a_mch = pi_mch.squeeze(-1).max(1)
                pi_agv, agv_pooled, _, _ = policy_agv(pt_ope=action_node,
                                                      hx_ope=hx,
                                                      mch_pooled=mch_pooled,
                                                      mask_mch_action=mask_mch_action,
                                                      CT_agv=env_CT_agv,
                                                      U_agv=env_U_agv,
                                                      greedy=True)

                _, a_agv = pi_agv.squeeze(-1).max(1)
                adj, fea, reward, done, ope_ids, mask_job, opes_min_ct, _, CT_mch, CT_job = env.step(
                    a_ope.cpu().numpy(),
                    a_mch,
                    a_agv,
                    gantt_chart)
                # rewards += reward
                j += 1
                if env.done():
                    # plt.savefig("./J{}M{}A{}makspan{}.svg".format(n_j, n_m, n_a, env.ET_mchs.max(-1).max(-1).max(-1)), format='svg', dpi=300, bbox_inches='tight')
                    plt.show()
                    break
            cost = env.ET_mchs.max(-1).max(-1)
            C_max.append(cost)
        return torch.tensor(cost)

    # make_spans.append(rewards - env.posRewards)
    # print(env.ST_mchs,env.mchsEndTimes,env.opIDsOnMchs)
    # print('REWARD',rewards - env.posRewards)
    total_cost = torch.cat(
        [eval_model_bat(bat, i, True if i == len(vali_set) - 1 else False) for i, bat in enumerate(vali_set)], 0)

    return total_cost
def get_travel_time():
    # tt=np.zeros((mch_num,mch_num))
    # tt=[[ 0,  4,  3,  3,  2,  2,  2,  4,  3,  3,  2,  4,],
    #      [ 4,  0,  2,  2,  3,  3,  3,  3,  3,  2,  3,  3,],
    #      [ 3,  4,  0,  3,  3,  2,  3,  2,  2,  4,  4,  4,],
    #      [ 3,  2,  3,  0,  2,  3,  2,  4,  3,  3,  3,  3,],
    #      [ 3,  2,  4,  4,  0,  4,  3,  4,  2,  2,  2,  3,],
    #      [ 3,  3,  4,  3,  3,  0,  2,  4,  2,  4,  4,  2,],
    #      [ 2,  4,  3,  3,  4,  2,  0,  2,  4,  3,  3,  2,],
    #      [ 3,  3,  3,  3,  4,  3,  2,  0,  4,  3,  4,  3,],
    #      [ 3,  3,  2,  2,  2,  3,  3,  3,  0,  2,  3,  4,],
    #      [ 2,  2,  3,  2,  4,  4,  3,  4,  4,  0,  4,  3,],
    #      [ 4,  4,  3,  2,  2,  3,  4,  4,  2,  3,  0,  2,],
    #      [ 2,  3,  4,  3,  3,  3,  3,  4,  4,  3,  3,  0,]]
    # tt = [[0, 12, 9, 8, 4, 4, 3, 11, 8, 9, 2, 12, ],
    #       [10, 0, 4, 4, 5, 7, 6, 5, 8, 3, 5, 6, ],
    #       [7, 10, 0, 7, 8, 2, 8, 4, 3, 11, 12, 10, ],
    #       [5, 3, 9, 0, 3, 7, 2, 11, 5, 9, 5, 7, ],
    #       [7, 4, 12, 10, 0, 11, 8, 11, 3, 4, 2, 5, ],
    #       [6, 5, 10, 6, 5, 0, 3, 10, 3, 12, 10, 4, ],
    #       [2, 10, 9, 9, 10, 3, 0, 3, 11, 8, 5, 3, ],
    #       [5, 5, 9, 8, 11, 7, 3, 0, 10, 8, 10, 7, ],
    #       [7, 6, 2, 3, 2, 8, 5, 7, 0, 4, 6, 10, ],
    #       [4, 3, 5, 4, 11, 10, 8, 11, 10, 0, 11, 7, ],
    #       [10, 11, 5, 3, 4, 6, 10, 11, 2, 7, 0, 4, ],
    #       [3, 5, 11, 5, 7, 9, 6, 12, 12, 5, 7, 0, ]]
    # tt = [[0, 7, 6, 5, 4, 4, 3, 6, 5, 6, 3, 7, ],
    #       [6, 0, 4, 4, 4, 5, 5, 4, 5, 4, 4, 4, ],
    #       [5, 6, 0, 5, 5, 3, 5, 4, 3, 7, 7, 6, ],
    #       [4, 3, 6, 0, 3, 5, 3, 7, 4, 6, 4, 5, ],
    #       [5, 4, 7, 6, 0, 7, 5, 7, 3, 4, 3, 4, ],
    #       [5, 4, 6, 4, 4, 0, 4, 6, 3, 7, 6, 4, ],
    #       [3, 6, 6, 6, 6, 3, 0, 3, 6, 5, 4, 3, ],
    #       [4, 4, 6, 6, 7, 5, 3, 0, 6, 5, 6, 5, ],
    #       [5, 5, 3, 3, 3, 6, 4, 5, 0, 4, 5, 6, ],
    #       [4, 3, 4, 4, 7, 6, 6, 6, 6, 0, 7, 5, ],
    #       [6, 7, 4, 3, 4, 5, 6, 6, 3, 5, 0, 4, ],
    #       [3, 4, 7, 4, 5, 6, 4, 7, 7, 4, 5, 0, ]]
    # tt = [
    #     [0, 6, 8, 6, 8, 10, 12, 10, 12],
    #     [8, 0, 2, 8, 2, 4, 6, 4, 6],
    #     [6, 10, 0, 10, 8, 2, 4, 6, 4],
    #     [12, 4, 6, 0, 6, 8, 10, 8, 10],
    #     [10, 2, 4, 6, 0, 6, 8, 2, 8],
    #     [8, 8, 2, 8, 6, 0, 6, 4, 2],
    #     [6, 10, 8, 10, 8, 6, 0, 6, 4],
    #     [12, 4, 6, 4, 2, 8, 10, 0, 10],
    #     [10, 6, 4, 6, 4, 2, 8, 2, 0]
    # ]
    tt= [
        [0, 6, 8, 10, 12],
        [12, 0, 6, 8, 10],
        [10, 6, 0, 6, 8],
        [8, 8, 6, 0, 6],
        [6, 10, 8, 6, 0]
    ]
    tt=[
        [0, 4, 6, 8, 6],
        [6, 0, 2, 4, 2],
        [8, 12, 0, 2, 4],
        [6, 10, 12, 0, 2],
        [4, 8, 10, 12, 0]
    ]
    tt=[
        [0, 2, 4, 10, 12],
        [12, 0, 2, 8, 10],
        [10, 12, 0, 6, 8],
        [4, 6, 8, 0, 2],
        [2, 4, 6, 12, 0]
    ]
    tt = [
        [0, 4, 8, 10, 14],
        [18, 0, 4, 6, 10],
        [20, 14, 0, 8, 6],
        [12, 8, 6, 0, 6],
        [14, 14, 12, 6, 0]
    ]
    tt = [
        [0, 7, 10, 8, 10, 9, 9, 9, 8, 10, 5, 7, 9, 10, 7, 4],
        [7, 0, 9, 9, 7, 10, 4, 6, 9, 4, 7, 5, 9, 10, 4, 9],
        [10, 9, 0, 6, 9, 7, 9, 5, 4, 5, 9, 5, 7, 9, 8, 5],
        [8, 9, 6, 0, 6, 8, 6, 4, 5, 5, 6, 7, 6, 8, 7, 9],
        [10, 7, 9, 6, 0, 7, 4, 9, 8, 9, 8, 10, 8, 5, 8, 5],
        [9, 10, 7, 8, 7, 0, 4, 8, 10, 6, 9, 4, 8, 7, 8, 8],
        [9, 4, 9, 6, 4, 4, 0, 10, 4, 8, 4, 4, 5, 8, 10, 9],
        [9, 6, 5, 4, 9, 8, 10, 0, 10, 7, 7, 4, 9, 4, 8, 7],
        [8, 9, 4, 5, 8, 10, 4, 10, 0, 7, 9, 5, 7, 6, 10, 6],
        [10, 4, 5, 5, 9, 6, 8, 7, 7, 0, 6, 10, 8, 10, 5, 7],
        [5, 7, 9, 6, 8, 9, 4, 7, 9, 6, 0, 10, 9, 5, 4, 5],
        [7, 5, 5, 7, 10, 4, 4, 4, 5, 10, 10, 0, 6, 4, 10, 8],
        [9, 9, 7, 6, 8, 8, 5, 9, 7, 8, 9, 6, 0, 8, 8, 4],
        [10, 10, 9, 8, 5, 7, 8, 4, 6, 10, 5, 4, 8, 0, 7, 4],
        [7, 4, 8, 7, 8, 8, 10, 8, 10, 5, 4, 10, 8, 7, 0, 7],
        [4, 9, 5, 9, 5, 8, 9, 7, 6, 7, 5, 8, 4, 4, 7, 0]
    ]
    # tt = np.zeros((15, 15))
    tt = np.array(tt)
    return tt

if __name__ == '__main__':

    from uniform_instance import uni_instance_gen, FJSPDataset
    import numpy as np
    import time
    import argparse
    from Params import configs

    parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
    parser.add_argument('--Pn_j', type=int, default=30, help='Number of jobs of instances to test')
    parser.add_argument('--Pn_m', type=int, default=20, help='Number of machines instances to test')
    parser.add_argument('--Pn_a', type=int, default=20, help='Number of AGVs instances to test')
    parser.add_argument('--Nn_j', type=int, default=30, help='Number of jobs on which to be loaded net are trained')
    parser.add_argument('--Nn_m', type=int, default=20, help='Number of machines on which to be loaded net are trained')
    parser.add_argument('--Nn_a', type=int, default=2, help='Number of AGVs on which to be loaded net are trained')
    parser.add_argument('--low', type=int, default=-99, help='LB of duration')
    parser.add_argument('--high', type=int, default=99, help='UB of duration')
    parser.add_argument('--seed', type=int, default=200, help='Cap seed for validate set generation')
    parser.add_argument('--n_vali', type=int, default=100, help='validation set size')
    params = parser.parse_args()


    from torch.utils.data import DataLoader
    from PPOwithValue import PPO
    import torch
    import os
    from torch.utils.data import Dataset

    def load_ppo(n_j_train=configs.n_j, n_j=configs.n_j, n_m=configs.n_m, n_a=configs.n_a, n_o=configs.n_m):
        ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
                  n_j=n_j,
                  n_m=n_m,
                  n_a=n_a,
                  n_o=n_o,
                  num_layers=configs.num_layers,
                  neighbor_pooling_type=configs.neighbor_pooling_type,
                  input_dim=configs.input_dim,
                  hidden_dim=configs.hidden_dim,
                  num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
                  num_mlp_layers_actor=configs.num_mlp_layers_actor,
                  hidden_dim_actor=configs.hidden_dim_actor,
                  num_mlp_layers_critic=configs.num_mlp_layers_critic,
                  hidden_dim_critic=configs.hidden_dim_critic)

        filepath = 'saved_network'
        filepath = os.path.join(filepath, 'FJSP_AGV_J{}M{}A{}'.format(10, 15, 5))
        filepath = os.path.join(filepath, 'best_value536.8816206760705')  # 8-8-2
        # filepath = os.path.join(filepath, 'best_value470.3091608129441')  # 8-8-2
        # filepath = os.path.join(filepath, 'best_value900.5218921229243')  # 20x5x2
        # filepath = os.path.join(filepath, 'best_value482.35984541699287')  # 8-8-2
        # filepath = os.path.join(filepath, 'best_value466.0101761803031')  # 8-8-2
        # filepath = os.path.join(filepath, 'best_value470.26522044092417')  # 8-8-2

        job_path = '{}.pth'.format('policy_job')
        mch_path = '{}.pth'.format('policy_mch')
        agv_path = '{}.pth'.format('policy_agv')

        job_path = os.path.join(filepath, job_path)
        mch_path = os.path.join(filepath, mch_path)
        agv_path = os.path.join(filepath, agv_path)

        ppo.policy_job.load_state_dict(torch.load(job_path))
        ppo.policy_mch.load_state_dict(torch.load(mch_path))
        ppo.policy_agv.load_state_dict(torch.load(agv_path))
        return ppo
    batch_size = 1
    SEEDs = [200]
    result = []
    load = True  # True：选择加载数据集， False：随机生成验证数据集
    vali_result_total = []
    for SEED in SEEDs:
        mean_makespan = []
        if load:
            # validate_dataset = np.load(file="FJSP_J%sM%s_unew_test_data.npy" % (configs.n_j, configs.n_m))
            # travel_time = load_travel_time("FJSPTinstances/DeroussiNorre/travel_time.txt")
            # travel_time = load_travel_time("FJSPTinstances/DeroussiNorre/travel_time.txt")
            # travel_time = np.array(travel_time)
            travel_time = get_travel_time()/2.0
            file_path = './FJSPTinstances/100x15x5'
            for instance in os.listdir(file_path):
                # num_operation每个工件的工序数量, deroussi_norre每个工序的加工时间
                deroussi_norre, num_operations = load_standard_instance('./'+file_path+ '/' + instance)
                # data = getdata('./FJSPTinstances/random_generated/Jobset01.txt')

                data_set = []
                last_ope_ids = []
                for i in range(batch_size):
                    data_set.append(deroussi_norre)
                    last_ope_ids.append([idx * max(num_operations) + i - 1 for idx, i in enumerate(num_operations)])
                data_set = np.array(data_set)
                # data_set = np.load("FJSPTinstances/DeroussiNorre/fjsp1.txt")
                num_job = len(data_set[0])
                # ppo = load_ppo(n_j=num_job, n_j_train=8, n_o=8)
                ppo = load_ppo(n_j=num_job, n_j_train=10, n_o=max(num_operations))
                valid_loader = DataLoader(data_set, batch_size=batch_size)
                t1 = time.time()
                # vali_result = validate(valid_loader, batch_size, ppo.policy_job, ppo.policy_mch, ppo.policy_agv, n_j=num_job, n_o=5, travel_time1=travel_time)
                vali_result = validate(valid_loader, batch_size, ppo.policy_job, ppo.policy_mch, ppo.policy_agv, n_j=num_job, n_o=max(num_operations), last_ope_ids=last_ope_ids, travel_time1=travel_time)
                t2 = time.time()
                vali_result_total.append(vali_result)
            print(vali_result_total)
            flattened_results = [item for sublist in vali_result_total for item in sublist]
            average_makespan = np.mean(flattened_results)
            print("Average makespan: ", average_makespan)                # np.save("OurInsJ%sM%sA%saveraMakespan%.2ftime%.2f" % (10, 8, 2, np.array(vali_result).mean(), t2 - t1), [i.item() for i in vali_result])
                # print(vali_result, t2-t1)
                # mean_makespan.append(vali_result)
            df = pd.DataFrame(flattened_results, columns=['Makespan'])
            df.to_excel('100x15x5.xlsx', index=False)

        else:
            # 测试的参数, 测试的时候param里面的参数也要修改，改成测试案例的规模
            N_JOBS_P = 50
            N_MACHINES_P = 20
            N_AGVS_P = 10
            num_val = 1
            validate_dataset = FJSPDataset(n_j=N_JOBS_P, n_m=N_MACHINES_P, low=configs.low, high=configs.high, num_samples=num_val, seed=SEED)
            # data = getdata('./FJSPTinstances/random_generated/Jobset01.txt')
            valid_loader = DataLoader(validate_dataset, batch_size=batch_size)
            ppo = load_ppo(n_j_train=25)
            t1 = time.time()
            vali_result = validate(valid_loader, batch_size, ppo.policy_job, ppo.policy_mch, ppo.policy_agv)
            t2 = time.time()
            np.save("OurInsJ%sM%sA%saveraMakespan%.2ftime%.2f" % (N_JOBS_P, N_MACHINES_P, N_AGVS_P, np.array(vali_result).mean(), t2 - t1), [i.item() for i in vali_result])
            # mean_makespan.append(vali_result)
            print(vali_result, np.array(vali_result).mean())

    # print(min(result))
