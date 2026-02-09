import torch.nn as nn
from models.mlp import MLPActor
from models.mlp import MLPCritic, MLP
import torch.nn.functional as F
from models.graphcnn_congForSJSSP import GraphCNN
from torch.distributions.categorical import Categorical
import torch
from Params import configs
from Mhattention import ProbAttention
from agent_utils import greedy_select_action, select_action_mch
from models.Pointer import Pointer

INIT = configs.Init


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs):
        # (batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)
        # (batch_size, 1 (unsqueezed), hidden_size)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)
        # 1st line of Eq.(3) in the paper
        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        return u_i


class Encoder(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, learn_eps, neighbor_pooling_type, device):
        super(Encoder, self).__init__()
        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)

    def forward(self, x, graph_pool, padded_nei, adj, ):
        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 adj=adj)

        return h_pooled, h_nodes


class Job_Actor(nn.Module):
    def __init__(self,
                 n_j,
                 n_m,
                 n_o,
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 device
                 ):
        super(Job_Actor, self).__init__()
        self.n_j = n_j
        self.device = device
        self.bn = torch.nn.BatchNorm1d(input_dim).to(device)
        self.n_m = n_m
        self.n_o = n_o
        self.n_ops_perjob = n_m
        self.device = device
        '''self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)'''
        self.encoder = Encoder(num_layers=num_layers,
                               num_mlp_layers=num_mlp_layers_feature_extract,
                               input_dim=input_dim,
                               hidden_dim=hidden_dim,
                               learn_eps=learn_eps,
                               neighbor_pooling_type=neighbor_pooling_type,
                               device=device).to(device)
        self._input = nn.Parameter(torch.Tensor(hidden_dim))
        self._input.data.uniform_(-1, 1).to(device)
        self.actor = MLPActor(3, hidden_dim * 4, hidden_dim, 1).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj,
                ope_ids,
                mask_job,
                mask_ope_mch,
                dur,
                a_index,
                old_action,
                mch_pooled,
                agv_pooled,
                old_policy=True,
                T=1,
                greedy=False
                ):
        h_pooled, h_nodes = self.encoder(x=x,
                                         graph_pool=graph_pool,
                                         padded_nei=padded_nei,
                                         adj=adj)

        if old_policy:
            dummy = ope_ids.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))  # unsqueeze(-1)用于张量在最后一个维度增加一个维度
            batch_node = h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)).to(self.device)
            fea_ope = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)
            # -----------------------------------------------------------------------------------------------------------
            # first_ope_ids_scores = self.actor(decoder_input, first_ope_ids_feature,0)
            h_pooled_repeated = h_pooled.unsqueeze(-2).expand_as(fea_ope)
            if mch_pooled == None:
                mch_pooled_repeated = self._input[None, None, :].expand_as(fea_ope).to(self.device)
            else:
                mch_pooled_repeated = mch_pooled.unsqueeze(-2).expand_as(fea_ope).to(self.device)
            if agv_pooled == None:
                agv_pooled_repeated = self._input[None, None, :].expand_as(fea_ope).to(self.device)
            else:
                agv_pooled_repeated = agv_pooled.unsqueeze(-2).expand_as(fea_ope).to(self.device)
            concateFea = torch.cat((fea_ope, h_pooled_repeated, mch_pooled_repeated, agv_pooled_repeated), dim=-1)
            ope_scores = self.actor(concateFea)  # 将10个工件的384个特征通过多层感知机变成一个score
            # ope_scores = self.attn(decoder_input, ope_feature)
            ope_scores = ope_scores * 10
            mask_job_reshaped = mask_job.reshape(ope_scores.size())
            ope_scores[mask_job_reshaped] = float('-inf')
            pi = F.softmax(ope_scores, dim=1)  # first_ope_ids_score经过softmax层后得到策略函数pi
            if greedy:
                a_ope, job_ids = greedy_select_action(pi, ope_ids)
                log_a = 0
            else:
                job_ids, log_a = select_action_mch(pi)  # a为选择工序的序号，index是选择工件的编号，log_a是对数概率密度函数
                a_ope = torch.ones(dummy.size(0), dtype=torch.int32)
                for i in range(job_ids.size(0)):
                    a_ = ope_ids[i][job_ids[i]]
                    a_ope[i] = a_
            a_ope1 = a_ope.type(torch.long).to(self.device)
            pt_ope_batch = dur.reshape(dummy.size(0), self.n_j * self.n_o, -1).to(self.device)  # 工序的加工时间
            mask_ope_mch = mask_ope_mch.reshape(dummy.size(0), -1, self.n_m)
            mask_mch_action = torch.gather(mask_ope_mch, 1, a_ope1.unsqueeze(-1).unsqueeze(-1) \
                                           .expand(mask_ope_mch.size(0), -1, mask_ope_mch.size(2)))  # 能加工选中的工序的机器序号为True
            # --------------------------------------------------------------------------------------------------------------------
            action_feature = torch.gather(batch_node, 1, a_ope1.unsqueeze(-1).unsqueeze(-1) \
                                          .expand(batch_node.size(0), -1, batch_node.size(2))).squeeze(1)
            pt_ope = torch.gather(pt_ope_batch, 1, a_ope1.unsqueeze(-1).unsqueeze(-1) \
                                  .expand(pt_ope_batch.size(0), -1, pt_ope_batch.size(2))).squeeze(1)  # [:,:-2]

            return a_ope, job_ids, log_a, pt_ope.detach(), action_feature.detach(), mask_mch_action.detach(), h_pooled.detach()
        else:
            dummy = ope_ids.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
            batch_node = h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)).to(self.device)
            fea_ope = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)
            # -----------------------------------------------------------------------------------------------------------
            # first_ope_ids_scores = self.actor(decoder_input, first_ope_ids_feature, 0)
            # first_ope_ids_scores = self.attn(h_pooled, first_ope_ids_feature)
            h_pooled_repeated = h_pooled.unsqueeze(-2).expand_as(fea_ope)
            if mch_pooled == None:
                mch_pooled_repeated = self._input[None, None, :].expand_as(fea_ope).to(self.device)
            else:
                mch_pooled_repeated = mch_pooled.unsqueeze(-2).expand_as(fea_ope).to(self.device)
            if agv_pooled == None:
                agv_pooled_repeated = self._input[None, None, :].expand_as(fea_ope).to(self.device)
            else:
                agv_pooled_repeated = agv_pooled.unsqueeze(-2).expand_as(fea_ope).to(self.device)
            concateFea = torch.cat((fea_ope, h_pooled_repeated, mch_pooled_repeated, agv_pooled_repeated), dim=-1)
            ope_scores = self.actor(concateFea)
            ope_scores = ope_scores.squeeze(-1) * 10
            mask_job_reshaped = mask_job.reshape(ope_scores.size())
            ope_scores[mask_job_reshaped] = float('-inf')
            pi = F.softmax(ope_scores, dim=1)

            dist = Categorical(pi)
            log_a = dist.log_prob(a_index.to(self.device))
            entropy = dist.entropy()

            a_ope1 = old_action.type(torch.long).to(self.device)
            pt_ope_batch = dur.reshape(dummy.size(0), self.n_j * self.n_m, -1).to(self.device)
            mask_ope_mch = mask_ope_mch.reshape(dummy.size(0), -1, self.n_m)
            mask_mch_action = torch.gather(mask_ope_mch, 1, a_ope1.unsqueeze(-1) \
                                           .unsqueeze(-1).expand(mask_ope_mch.size(0), -1, mask_ope_mch.size(2)))
            # --------------------------------------------------------------------------------------------------------------------
            action_feature = torch.gather(batch_node, 1, a_ope1.unsqueeze(-1) \
                                          .unsqueeze(-1).expand(batch_node.size(0), -1, batch_node.size(2))).squeeze(1)
            pt_ope = torch.gather(pt_ope_batch, 1, a_ope1.unsqueeze(-1).unsqueeze(-1) \
                                  .expand(pt_ope_batch.size(0), -1, pt_ope_batch.size(2))).squeeze(1)
            v = self.critic(h_pooled)
            return entropy, v, log_a, pt_ope.detach(), action_feature.detach(), mask_mch_action.detach(), h_pooled.detach()


class Mch_Actor(nn.Module):
    def __init__(self, n_j,
                 n_m,
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 device):
        super(Mch_Actor, self).__init__()
        self.n_j = n_j
        self.bn = torch.nn.BatchNorm1d(hidden_dim).to(device)
        # machine size for problems, no business with network
        self.n_m = n_m
        self.hidden_size = hidden_dim
        self.n_ops_perjob = n_m
        self.device = device
        self.fc = nn.Linear(2, hidden_dim, bias=False).to(device)
        self._input = nn.Parameter(torch.Tensor(hidden_dim))
        self._input.data.uniform_(-1, 1).to(device)
        self.actor = MLPActor(3, hidden_dim * 4, hidden_dim, 1).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim * n_m * 4, hidden_dim_critic, 1).to(device)
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, pt_ope, hx_ope, agv_pooled, mask_mch_action, CT_mch, a_mch=None, last_hh=None, old_policy=True,
                greedy=False):
        CT_mch = CT_mch / configs.et_normalize_coef
        pt_ope_norm = pt_ope / configs.et_normalize_coef  # 工序在每个机器上的加工时间
        feature = torch.cat((CT_mch.unsqueeze(-1), pt_ope_norm.unsqueeze(-1)), -1)  # 机器现在已经加工的时长？和工序在每个机器的加工时间
        fea_mch = self.bn(self.fc(feature).reshape(-1, self.hidden_size))\
            .reshape(-1, self.n_m, self.hidden_size)  # 全连接层+批量归一化
        fea_mch_pooled = fea_mch.mean(dim=1)
        h_pooled_repeated = fea_mch_pooled.unsqueeze(1).expand_as(fea_mch)
        hx_ope_re = hx_ope.unsqueeze(1).expand_as(fea_mch)
        if agv_pooled == None:
            agv_pooled_repeated = self._input[None, None, :].expand_as(fea_mch).to(self.device)
        else:
            agv_pooled_repeated = agv_pooled.unsqueeze(1).expand_as(fea_mch).to(self.device)

        concateFea = torch.cat((fea_mch, hx_ope_re, h_pooled_repeated, agv_pooled_repeated), dim=-1)
        mch_scores1 = self.actor(concateFea)  # 将全部特征输入到MLP解码器中，得到工序选择每个机器的分数
        mch_scores = mch_scores1.squeeze(-1) * 10  # squeeze(-1)将最后一维去掉
        mch_scores = mch_scores.masked_fill(mask_mch_action.squeeze(1).bool(), float("-inf"))  # 被mask遮住的机器分数为-inf
        pi_mch = F.softmax(mch_scores, dim=1)

        if old_policy:
            if greedy:
                _, a_mch = pi_mch.squeeze(-1).max(1)
                log_mch = 0
            else:
                a_mch, log_mch = select_action_mch(pi_mch)  # a为选择工序的序号，index是选择工件的编号，log_a是对数概率密度函数
            return pi_mch, fea_mch_pooled, a_mch, log_mch
        else:
            dist = Categorical(pi_mch)
            log_mch = dist.log_prob(a_mch.to(self.device))
            entropy = dist.entropy()
            transfer_con = concateFea.reshape(concateFea.size()[0], -1)
            v = self.critic(transfer_con)
            return pi_mch, fea_mch_pooled, [entropy, v], log_mch



class Agv_Actor(nn.Module):
    def __init__(self,
                 n_a,
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 device):
        super(Agv_Actor, self).__init__()
        self.bn = torch.nn.BatchNorm1d(hidden_dim).to(device)
        self.n_a = n_a
        self.hidden_size = hidden_dim
        self.n_ops_perjob = n_a
        self.device = device

        self.fc = nn.Linear(2, hidden_dim, bias=False).to(device)
        self._input = nn.Parameter(torch.Tensor(hidden_dim))
        self._input.data.uniform_(-1, 1).to(device)
        self.actor = MLPActor(3, hidden_dim * 4, hidden_dim, 1).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim * n_a * 4, hidden_dim_critic, 1).to(device)
        if INIT:
            for name, p in self.named_parameters():
                if 'weight' in name:
                    if len(p.size()) >= 2:
                        nn.init.orthogonal_(p, gain=1)
                elif 'bias' in name:
                    nn.init.constant_(p, 0)

    def forward(self, pt_ope, hx_ope, mch_pooled, mask_mch_action, CT_agv, U_agv, a_agv=None, last_hh=None, old_policy=True, greedy=False):
        CT_agv = CT_agv / configs.et_normalize_coef  # fixme configs.et_normalize_coef 要吗
        fea_agv = torch.cat((CT_agv.unsqueeze(-1), U_agv.unsqueeze(-1)), -1)
        fea_agv_bn = self.bn(self.fc(fea_agv).reshape(-1, self.hidden_size)) \
            .reshape(-1, self.n_a, self.hidden_size)  # 全连接层+批量归一化
        fea_agv_pooled = fea_agv_bn.mean(dim=1)
        h_pooled_repeated = fea_agv_pooled.unsqueeze(1).expand_as(fea_agv_bn)
        hx_ope_re = hx_ope.unsqueeze(1).expand_as(fea_agv_bn)
        mch_pooled_repeated = mch_pooled.unsqueeze(1).expand_as(fea_agv_bn).to(self.device)
        concateFea = torch.cat((fea_agv_bn, hx_ope_re, mch_pooled_repeated, h_pooled_repeated), dim=-1)
        agv_scores = self.actor(concateFea)  # 将全部特征输入到MLP解码器中，得到工序选择每个机器的分数
        agv_scores1 = agv_scores.squeeze(-1) * 10  # squeeze(-1)将最后一维去掉
        pi_agv = F.softmax(agv_scores1, dim=1)

        if old_policy:
            if greedy:
                _, a_agv = pi_agv.squeeze(-1).max(1)
                log_agv = 0
            else:
                a_agv, log_agv = select_action_mch(pi_agv)  # a为选择agv的序号，index是选择工件的编号，log_a是对数概率密度函数
            return pi_agv, fea_agv_pooled, a_agv, log_agv
        else:
            dist = Categorical(pi_agv)
            log_agv = dist.log_prob(a_agv.to(self.device))
            entropy = dist.entropy()
            transfer_con = concateFea.reshape(concateFea.size()[0], -1)
            v = self.critic(transfer_con)
            return pi_agv, fea_agv_pooled, [entropy, v], log_agv


if __name__ == '__main__':
    print('Go home')
