from matplotlib import pyplot as plt

from FJSPT_Env import FJSPTGanttChart
from mb_agg import *
from agent_utils import eval_actions
from models.PPO_Actor import Job_Actor, Mch_Actor, Agv_Actor
from copy import deepcopy
import torch
import time
from torch.distributions.categorical import Categorical
import torch.nn as nn
import numpy as np
from Params import configs
from validation import validate
from epsGreedyForMch import PredictMch
import os

device = torch.device(configs.device)
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR


class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.ope_ids_mb = []
        self.mask_job = []
        self.r_mb = []
        self.done_mb = []
        self.job_logprobs = []
        self.mch_logprobs = []
        self.agv_logprobs = []
        self.mask_ope_mch = []
        self.first_ope = []
        self.pre_ope = []
        self.a_job = []
        self.a_ope = []
        self.a_mch = []
        self.a_agv = []
        self.dur = []
        self.CT_mch = []
        self.CT_agv = []
        self.U_agv = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.ope_ids_mb[:]
        del self.mask_job[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.job_logprobs[:]
        del self.mch_logprobs[:]
        del self.agv_logprobs[:]
        del self.mask_ope_mch[:]
        del self.first_ope[:]
        del self.pre_ope[:]
        del self.a_job[:]
        del self.a_ope[:]
        del self.a_mch[:]
        del self.a_agv[:]
        del self.dur[:]
        del self.CT_mch[:]
        del self.CT_agv[:]
        del self.U_agv[:]


def initWeights(net, scheme='orthogonal'):
    for e in net.parameters():
        if scheme == 'orthogonal':
            if len(e.size()) >= 2:
                nn.init.orthogonal_(e)
        elif scheme == 'normal':
            nn.init.normal(e, std=1e-2)
        elif scheme == 'xavier':
            nn.init.xavier_normal(e)


def adv_normalize(adv):
    std = adv.std()
    assert std != 0. and not torch.isnan(std), 'Need nonzero std'
    n_advs = (adv - adv.mean()) / (adv.std() + 1e-8)
    return n_advs


class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_j,
                 n_m,
                 n_a,
                 n_o,
                 num_layers,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy_job = Job_Actor(n_j=n_j,
                                    n_m=n_m,
                                    n_o=n_o,
                                    num_layers=num_layers,
                                    learn_eps=False,
                                    neighbor_pooling_type=neighbor_pooling_type,
                                    input_dim=input_dim,
                                    hidden_dim=hidden_dim,
                                    num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                    num_mlp_layers_critic=num_mlp_layers_critic,
                                    hidden_dim_critic=hidden_dim_critic,
                                    device=device)

        self.policy_mch = Mch_Actor(n_j=n_j,
                                    n_m=n_m,
                                    num_layers=num_layers,
                                    learn_eps=False,
                                    neighbor_pooling_type=neighbor_pooling_type,
                                    input_dim=input_dim,
                                    hidden_dim=hidden_dim,
                                    num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                    num_mlp_layers_critic=num_mlp_layers_critic,
                                    hidden_dim_critic=hidden_dim_critic,
                                    device=device)

        self.policy_agv = Agv_Actor(n_a=n_a,
                                    num_layers=num_layers,
                                    learn_eps=False,
                                    neighbor_pooling_type=neighbor_pooling_type,
                                    input_dim=input_dim,
                                    hidden_dim=hidden_dim,
                                    num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                    num_mlp_layers_critic=num_mlp_layers_critic,
                                    hidden_dim_critic=hidden_dim_critic,
                                    device=device)

        self.policy_old_job = deepcopy(self.policy_job)
        self.policy_old_mch = deepcopy(self.policy_mch)
        self.policy_old_agv = deepcopy(self.policy_agv)
        self.policy_old_job.load_state_dict(self.policy_job.state_dict())
        self.policy_old_mch.load_state_dict(self.policy_mch.state_dict())
        self.policy_old_agv.load_state_dict(self.policy_agv.state_dict())
        self.job_optimizer = torch.optim.Adam(self.policy_job.parameters(), lr=lr)
        self.mch_optimizer = torch.optim.Adam(self.policy_mch.parameters(), lr=lr)
        self.agv_optimizer = torch.optim.Adam(self.policy_agv.parameters(), lr=lr)
        self.job_scheduler = torch.optim.lr_scheduler.StepLR(self.job_optimizer,
                                                             step_size=configs.decay_step_size,
                                                             gamma=configs.decay_ratio)
        self.mch_scheduler = torch.optim.lr_scheduler.StepLR(self.mch_optimizer,
                                                             step_size=configs.decay_step_size,
                                                             gamma=configs.decay_ratio)
        self.agv_scheduler = torch.optim.lr_scheduler.StepLR(self.agv_optimizer,
                                                             step_size=configs.decay_step_size,  # 每训练step_size个epoch，更新一次参数
                                                             gamma=configs.decay_ratio)  # 新的lr=gamma * lr
        self.MSE = nn.MSELoss()

    def update(self, memories):
        """
        更新ppo的网络参数
        :param memories:
        :return:
        """
        ploss_coef = configs.ploss_coef  # -1 裁剪代理损失占比系数
        vloss_coef = configs.vloss_coef  # 0.5 critic损失的占比系数
        entloss_coef = configs.entloss_coef  # 1  熵损失的占比系数
        rewards_all_env = []
        for i in range(configs.batch_size):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed((memories.r_mb[0][i]).tolist()),
                                           reversed(memories.done_mb[0][i].tolist())):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            rewards_all_env.append(rewards)
        rewards_all_env = torch.stack(rewards_all_env, 0)
        for _ in range(configs.k_epochs):
            g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                                     batch_size=torch.Size(
                                         [configs.batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                                     n_nodes=configs.n_j * configs.n_m,
                                     device=device)
            job_log_prob = []
            mch_log_prob = []
            agv_log_prob = []
            vals_job = []
            vals_mch = []
            vals_agv = []
            job_entropies = []
            mch_entropies = []
            agv_entropies = []
            job_log_old_prob = memories.job_logprobs[0]
            mch_log_old_prob = memories.mch_logprobs[0]
            agv_log_old_prob = memories.agv_logprobs[0]
            env_mask_ope_mch = memories.mask_ope_mch[0]
            env_dur = memories.dur[0]
            mch_pooled = None
            agv_pooled = None
            for i in range(len(memories.fea_mb)):
                env_fea = memories.fea_mb[i]
                env_adj = memories.adj_mb[i]
                env_first_ope_ids = memories.ope_ids_mb[i]
                env_mask = memories.mask_job[i]
                a_job = memories.a_job[i]
                env_CT_mch = memories.CT_mch[i]
                env_CT_agv = memories.CT_agv[i]
                env_U_agv = memories.U_agv[i]
                old_ope = memories.a_ope[i]
                old_mch = memories.a_mch[i]
                old_agv = memories.a_agv[i]
                job_entropy, val_job, log_a, action_node, _, mask_mch_action, hx = self.policy_job(x=env_fea,
                                                                                                   graph_pool=g_pool_step,
                                                                                                   padded_nei=None,
                                                                                                   adj=env_adj,
                                                                                                   ope_ids=env_first_ope_ids,
                                                                                                   mask_job=env_mask,
                                                                                                   mask_ope_mch=env_mask_ope_mch,
                                                                                                   dur=env_dur,
                                                                                                   a_index=a_job,
                                                                                                   old_action=old_ope,
                                                                                                   mch_pooled=mch_pooled,
                                                                                                   agv_pooled=agv_pooled,
                                                                                                   old_policy=False
                                                                                                   )
                pi_mch, mch_pooled, [mch_entropy, val_mch], log_mch = self.policy_mch(pt_ope=action_node,
                                                                                      hx_ope=hx,
                                                                                      agv_pooled=agv_pooled,
                                                                                      mask_mch_action=mask_mch_action,
                                                                                      CT_mch=env_CT_mch,
                                                                                      a_mch=old_mch,
                                                                                      old_policy=False)
                pi_agv, agv_pooled, [agv_entropy, val_agv], log_agv = self.policy_agv(pt_ope=action_node,
                                                                                      hx_ope=hx,
                                                                                      mch_pooled=mch_pooled,
                                                                                      mask_mch_action=mask_mch_action,
                                                                                      CT_agv=env_CT_agv,
                                                                                      U_agv=env_U_agv,
                                                                                      a_agv=old_agv,
                                                                                      old_policy=False)
                vals_job.append(val_job)
                vals_mch.append(val_mch)
                vals_agv.append(val_agv)

                job_entropies.append(job_entropy)
                mch_entropies.append(mch_entropy)
                agv_entropies.append(agv_entropy)

                job_log_prob.append(log_a)
                mch_log_prob.append(log_mch)
                agv_log_prob.append(log_agv)

            job_log_prob, job_log_old_prob = torch.stack(job_log_prob, 0).permute(1, 0), torch.stack(job_log_old_prob,
                                                                                                     0).permute(1, 0)
            mch_log_prob, mch_log_old_prob = torch.stack(mch_log_prob, 0).permute(1, 0), torch.stack(mch_log_old_prob,
                                                                                                     0).permute(1, 0)
            agv_log_prob, agv_log_old_prob = torch.stack(agv_log_prob, 0).permute(1, 0), torch.stack(agv_log_old_prob,
                                                                                                     0).permute(1, 0)
            vals_job = torch.stack(vals_job, 0).squeeze(-1).permute(1, 0)
            vals_mch = torch.stack(vals_mch, 0).squeeze(-1).permute(1, 0)
            vals_agv = torch.stack(vals_agv, 0).squeeze(-1).permute(1, 0)
            job_entropies = torch.stack(job_entropies, 0).permute(1, 0)
            mch_entropies = torch.stack(mch_entropies, 0).permute(1, 0)
            agv_entropies = torch.stack(agv_entropies, 0).permute(1, 0)

            job_loss_sum = 0
            mch_loss_sum = 0
            agv_loss_sum = 0
            for j in range(configs.batch_size):
                job_ratios = torch.exp(job_log_prob[j] - job_log_old_prob[j].detach())
                mch_ratios = torch.exp(mch_log_prob[j] - mch_log_old_prob[j].detach())
                agv_ratios = torch.exp(agv_log_prob[j] - agv_log_old_prob[j].detach())

                advantages_job = rewards_all_env[j] - vals_job[j].detach()
                advantages_job = adv_normalize(advantages_job)
                job_surr1 = job_ratios * advantages_job
                job_surr2 = torch.clamp(job_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_job
                job_v_loss = self.MSE(vals_job[j], rewards_all_env[j])
                job_p_loss = - torch.min(job_surr1, job_surr2)  # PPO本身是想要增大优势函数，在损失函数里给负值可以使优势函数朝增大的方向发展
                job_loss = ploss_coef * job_p_loss + vloss_coef * job_v_loss - entloss_coef * job_entropies[j]  # c_p, c_v, c_e
                job_loss_sum += job_loss

                advantages_mch = rewards_all_env[j] - vals_mch[j].detach()
                advantages_mch = adv_normalize(advantages_mch)
                mch_surr1 = mch_ratios * advantages_mch
                mch_surr2 = torch.clamp(mch_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_mch
                mch_v_loss = self.MSE(vals_mch[j], rewards_all_env[j])
                mch_p_loss = - torch.min(mch_surr1, mch_surr2)
                mch_loss = ploss_coef * mch_p_loss + vloss_coef * mch_v_loss - entloss_coef * mch_entropies[j]
                mch_loss_sum += mch_loss

                advantages_agv = rewards_all_env[j] - vals_agv[j].detach()
                advantages_agv = adv_normalize(advantages_agv)
                agv_surr1 = agv_ratios * advantages_agv
                agv_surr2 = torch.clamp(agv_ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_agv
                agv_v_loss = self.MSE(vals_agv[j], rewards_all_env[j])
                agv_p_loss = - torch.min(agv_surr1, agv_surr2)
                agv_loss = ploss_coef * agv_p_loss + vloss_coef * agv_v_loss - entloss_coef * agv_entropies[j] * 2
                agv_loss_sum += agv_loss

            self.job_optimizer.zero_grad()
            job_loss_sum.mean().backward(retain_graph=True)
            self.mch_optimizer.zero_grad()
            mch_loss_sum.mean().backward(retain_graph=True)
            self.agv_optimizer.zero_grad()
            agv_loss_sum.mean().backward(retain_graph=True)

            self.job_optimizer.step()
            self.mch_optimizer.step()
            self.agv_optimizer.step()

        # Copy new weights into old policy:
        self.policy_old_job.load_state_dict(self.policy_job.state_dict())
        self.policy_old_mch.load_state_dict(self.policy_mch.state_dict())
        self.policy_old_agv.load_state_dict(self.policy_agv.state_dict())
        if configs.decayflag:  # 是否更新学习率
            self.job_scheduler.step()
            self.mch_scheduler.step()
            self.agv_scheduler.step()
        return job_loss_sum.mean().item(), mch_loss_sum.mean().item(), agv_loss_sum.mean().item()


def main(epochs):
    from uniform_instance import FJSPDataset
    from FJSPT_Env import FJSPT

    log = []
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size(
                                 [configs.batch_size, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
                             n_nodes=configs.n_j * configs.n_m,
                             device=device)
    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
              n_j=configs.n_j,
              n_m=configs.n_m,
              n_a=configs.n_a,
              n_o=configs.n_m,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)
    train_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, configs.num_ins, 200)  # 12800个训练样本（每个样本10个工件5个机器，每个工件5道工序）
    validate_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, 128, 200)  # 生成128个样本进行测试，生成加工时间
    data_loader = DataLoader(train_dataset, batch_size=configs.batch_size)
    valid_loader = DataLoader(validate_dataset, batch_size=configs.batch_size)
    record = 1000000
    for epoch in range(epochs):
        memory = Memory()
        ppo.policy_old_job.train()  # .train()表示模型处于训练模式
        ppo.policy_old_mch.train()
        ppo.policy_old_agv.train()
        times, losses, rewards2, critic_rewards = [], [], [], []
        start = time.time()
        costs = []
        losses, rewards, critic_loss = [], [], []
        gantt_chart = FJSPTGanttChart(configs.n_j, configs.n_m, configs.n_a)
        for batch_idx, batch in enumerate(data_loader):
            env = FJSPT(configs.n_j, configs.n_m, configs.n_a, configs.n_m)
            data = batch.numpy()
            adj, fea, ope_ids, mask_job, mask_ope_mch, dur, CT_mch, CT_job, CT_agv, U_agv = env.reset(data)
            job_log_prob = []
            mch_log_prob = []
            agv_log_prob = []
            r_mb = []
            done_mb = []
            first_ope = []  # 第一个工序
            pre_ope = []  # 前一个工序
            j = 0
            a_mch = None
            a_agv = None
            last_hh = None
            fea_mch_pooled = None
            fea_agv_pooled = None
            ep_rewards = - env.initQuality
            env_dur = torch.from_numpy(np.copy(dur)).float().to(device)
            while True:
                env_adj = aggr_obs(deepcopy(adj).to(device).to_sparse(), configs.n_j * configs.n_m)  # to_sparse转为稀疏张量
                env_fea = torch.from_numpy(np.copy(fea)).float().to(device)
                env_fea = deepcopy(env_fea).reshape(-1, env_fea.size(-1))
                env_ope_ids = torch.from_numpy(np.copy(ope_ids)).long().to(device)
                env_mask_job = torch.from_numpy(np.copy(mask_job)).to(device)  # 工件加工完成对应的位置为True
                env_CT_mch = torch.from_numpy(np.copy(CT_mch)).float().to(device)
                env_CT_agv = torch.from_numpy(np.copy(CT_agv)).float().to(device)
                env_U_agv = torch.from_numpy(np.copy(U_agv)).float().to(device)
                env_mask_ope_mch = torch.from_numpy(np.copy(mask_ope_mch)).to(device)
                # 工件策略选择智能体，pt_ope为工序在每个机器上的加工时间
                a_ope, job_idx, log_ope, pt_ope, _, mask_mch_action, hx_ope = ppo.policy_old_job(x=env_fea,
                                                                                                 graph_pool=g_pool_step,
                                                                                                 padded_nei=None,
                                                                                                 adj=env_adj,
                                                                                                 ope_ids=env_ope_ids,
                                                                                                 mask_job=env_mask_job,
                                                                                                 mask_ope_mch=env_mask_ope_mch,
                                                                                                 dur=env_dur,
                                                                                                 a_index=0,
                                                                                                 old_action=0,
                                                                                                 mch_pooled=fea_mch_pooled,
                                                                                                 agv_pooled=fea_agv_pooled
                                                                                                 )
                # 机器策略选择智能体
                pi_mch, fea_mch_pooled, a_mch, log_mch = ppo.policy_old_mch(pt_ope=pt_ope,
                                                                            hx_ope=hx_ope,
                                                                            agv_pooled=fea_agv_pooled,
                                                                            mask_mch_action=mask_mch_action,
                                                                            CT_mch=env_CT_mch,
                                                                            a_mch=a_mch,
                                                                            last_hh=last_hh)
                # Agv策略选择智能体
                pi_agv, fea_agv_pooled, a_agv, log_agv = ppo.policy_old_agv(pt_ope=pt_ope,
                                                                            hx_ope=hx_ope,
                                                                            mch_pooled=fea_mch_pooled,
                                                                            mask_mch_action=mask_mch_action,
                                                                            CT_agv=env_CT_agv,
                                                                            U_agv=env_U_agv,
                                                                            a_agv=a_agv,
                                                                            last_hh=last_hh)

                job_log_prob.append(log_ope)
                mch_log_prob.append(log_mch)
                agv_log_prob.append(log_agv)
                memory.fea_mb.append(env_fea)
                memory.adj_mb.append(env_adj)
                memory.ope_ids_mb.append(env_ope_ids)
                memory.mask_job.append(env_mask_job)
                memory.a_job.append(job_idx)
                memory.CT_mch.append(env_CT_mch)
                memory.CT_agv.append(env_CT_agv)
                memory.U_agv.append(env_U_agv)
                memory.a_ope.append(deepcopy(a_ope).to(device))
                memory.a_mch.append(a_mch)
                memory.a_agv.append(a_agv)
                memory.mask_ope_mch.append(env_mask_ope_mch)

                memory.pre_ope.append(pre_ope)
                adj, fea, reward, done, ope_ids, mask_job, opes_min_ct, mask_ope_mch, CT_mch, CT_job = env.step(a_ope.cpu().numpy(), a_mch, a_agv)
                CT_agv, U_agv = env.CT_agv, env.U_agv
                ep_rewards += reward
                r_mb.append(deepcopy(reward))
                done_mb.append(deepcopy(done))
                j += 1
                if env.done():
                    # plt.show()
                    plt.close()
                    break
            memory.dur.append(env_dur)
            memory.first_ope.append(first_ope)
            memory.job_logprobs.append(job_log_prob)
            memory.mch_logprobs.append(mch_log_prob)
            memory.agv_logprobs.append(agv_log_prob)
            memory.r_mb.append(torch.tensor(r_mb).float().permute(1, 0))
            memory.done_mb.append(torch.tensor(done_mb).float().permute(1, 0))
            ep_rewards -= env.posRewards
            loss_job, loss_mch, loss_agv = ppo.update(memory)
            memory.clear_memory()
            mean_reward = np.mean(ep_rewards)
            log.append([batch_idx, mean_reward])
            if batch_idx % 100 == 0:
                file_writing_obj = open('./log/' + 'log_' + str(configs.n_j) + '_' + str(configs.n_m) + '_' \
                                        + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                file_writing_obj.write(str(log))
            rewards.append(np.mean(ep_rewards).item())
            losses.append([loss_job, loss_mch, loss_agv])
            critic_loss.append(loss_job)
            cost = env.ET_mchs.max(-1).max(-1)  # 这个应该就是完工时间=env.CT_mch
            costs.append(cost.mean())
            step = 20
            filepath = 'saved_network'
            if (batch_idx + 1) % step == 0:
                end = time.time()
                times.append(end - start)
                start = end
                mean_loss = np.mean(losses[-step:], 0)
                mean_reward = np.mean(costs[-step:])
                critic_losses = np.mean(critic_loss[-step:])
                filename = 'FJSP_AGV_{}'.format('J' + str(configs.n_j) + 'M' + str(configs.n_m) + "A" + str(configs.n_a))
                filepath = os.path.join(filepath, filename)
                epoch_dir = os.path.join(filepath, '%s_%s' % (100, batch_idx))
                if not os.path.exists(epoch_dir):
                    os.makedirs(epoch_dir)
                job_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_job'))
                machine_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_mch'))
                agv_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_agv'))
                torch.save(ppo.policy_job.state_dict(), job_savePath)
                torch.save(ppo.policy_mch.state_dict(), machine_savePath)
                torch.save(ppo.policy_agv.state_dict(), agv_savePath)
                print('Batch %d/%d, reward: %2.3f, job loss: %2.3f, mch loss: %2.3f, agv loss: %2.4f, took: %2.4fs' % \
                      (batch_idx, len(data_loader), mean_reward, mean_loss[0], mean_loss[1], mean_loss[2], times[-1]))
                validation_log = validate(valid_loader, configs.batch_size, ppo.policy_job, ppo.policy_mch, ppo.policy_agv)
                validation_log = validation_log.mean()
                if validation_log < record:
                    epoch_dir = os.path.join(filepath, 'best_value{}'.format(validation_log))
                    if not os.path.exists(epoch_dir):
                        os.makedirs(epoch_dir)
                    job_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_job'))
                    machine_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_mch'))
                    agv_savePath = os.path.join(epoch_dir, '{}.pth'.format('policy_agv'))
                    torch.save(ppo.policy_job.state_dict(), job_savePath)
                    torch.save(ppo.policy_mch.state_dict(), machine_savePath)
                    torch.save(ppo.policy_agv.state_dict(), agv_savePath)
                    record = validation_log
                print('The validation quality is:', validation_log.item())
                file_writing_obj1 = open('./log/' + 'vali_' + str(configs.n_j) + '_' + str(configs.n_m) + \
                                         '_' + str(configs.low) + '_' + str(configs.high) + '.txt', 'w')
                file_writing_obj1.write(str(validation_log))
        np.savetxt('./log/J_{}_M{}_A{}_makespan{}.txt'.format(configs.n_j, configs.n_m, configs.n_a, record), costs, delimiter="\n")
        torch.cuda.empty_cache()
        plt.figure()
        plt.plot(costs)
        plt.show()
    print("128个测试平均值最好结果为：{}".format(record))


if __name__ == '__main__':
    total1 = time.time()
    main(1)
    total2 = time.time()
    print("total time consumption: {}".format(total2 - total1))
