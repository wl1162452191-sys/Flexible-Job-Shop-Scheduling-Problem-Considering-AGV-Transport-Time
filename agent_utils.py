from torch.distributions.categorical import Categorical

import torch


def select_action(p, cadidate, memory, log_prob):
    dist = Categorical(p.squeeze())
    s = dist.sample()

    if memory is not None: log_prob.append(dist.log_prob(s).cpu().tolist())
    action = []
    for i in range(s.size(0)):
        a = cadidate[i][s[i]].cpu().tolist()
        action.append(a)

    return action, s


def select_action_ope(pi, ope_ids):
    """

    :param pi: 强化学习的策略函数pi
    :param ope_ids: 工序序号，工件加工到第几个工序
    :return: 选中的工序，选中的工件序号，对应的概率
    """
    dist = Categorical(pi.squeeze())
    s = dist.sample()  # s为根据策略pi抽样得到的candidate的序号
    action = []
    log_a = dist.log_prob(s)  # 对数概率密度函数
    for i in range(s.size(0)):
        a = ope_ids[i][s[i]]
        action.append(a)
    action = torch.stack(action, 0)  # 将张量action沿着维度0进行堆叠
    return action, s, log_a


def select_action_mch(p):
    """
    根据概率p选择机器，s为选择的机器的编号
    :param p:
    :return:
    """
    dist = Categorical(p.squeeze())
    s = dist.sample()
    log_a = dist.log_prob(s)
    return s, log_a


def select_action_agv(p):
    """
    根据概率p选择agv，s为选择的机器的编号
    :param p:
    :return:
    """
    dist = Categorical(p.squeeze())
    s = dist.sample()
    log_a = dist.log_prob(s)
    return s, log_a


# evaluate the actions
def eval_actions(p, actions):
    softmax_dist = Categorical(p)
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


# select action method for test
def greedy_select_action(p, candidate):
    _, index = p.squeeze(-1).max(1)
    action = []
    for i in range(index.size(0)):
        a = candidate[i][index[i]]
        action.append(a)
    action = torch.stack(action, 0)
    return action, index


# select action method for test
def sample_select_action(p, candidate):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    return candidate[s]
