# https://github.com/hijkzzz/reinforcement-learning.pytorch/blob/master/src/sac.py

import random
import copy
import pandas as pd
import matplotlib.pyplot as plt

from collections import deque
import os
import numpy as np
import torch
from torch import nn
from torch.nn import init, Module, functional as F
from torch.nn.modules import Linear

import torch_utils as tu

# # #### profiler start ####
import builtins

try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func
    builtins.profile = profile
# # #### profiler end ####


class Configs:
    def __init__(self, name):
        self.name = name
        self.lr = 5e-3
        self.num_epochs = 30000
        self.base_i0 = 1000
        self.n_samples = 200
        self.k_days = 5
        self.label_days = 5
        self.strategy_days = 250
        self.adaptive_count = 5
        self.adaptive_lrx = 10 # learning rate * 배수
        self.es_max_count = 50
        self.retrain_days = 240
        self.test_days = 480  # test days
        self.init_train_len = 500
        self.normalizing_window = 500  # norm windows for macro data
        self.use_accum_data = True # [sampler] 데이터 누적할지 말지
        self.adaptive_flag = True
        self.n_pretrain = 20

        self.datatype = 'app'

        self.cost_rate = 0.003
        self.plot_freq = 1000

        self.init()
        self.set_path()

    def init(self):
        self.init_weight()
        self.init_loop()

    def init_weight(self):
        if self.datatype == 'app':
            self.cash_idx = 3
            self.base_weight = [0.7, 0.2, 0.1, 0.0]
            # self.base_weight = None
        else:
            self.cash_idx = 0
            self.base_weight = None

    def set_path(self):
        self.outpath = './out/{}/'.format(self.name)
        os.makedirs(self.outpath, exist_ok=True)

    def init_loop(self):
        adaptive_flag = self.adaptive_flag
        if adaptive_flag:
            self.es_max = -1
        else:
            self.es_max = self.es_max_count

        self.min_eval_loss = 999999
        self.es_count = 0
        self.loss_wgt = {'logy_pf': 1., 'mdd_pf': 1., 'logy': 1., 'wgt': 0., 'wgt2': 1., 'wgt_guide': 0., 'cost': 0., 'entropy': 0.}
        self.adaptive_loss_wgt = {'logy_pf': -1., 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.1, 'cost': 1., 'entropy': -1.}

        return adaptive_flag

    def export(self):
        return_str = ""
        for key in self.__dict__.keys():
            return_str += "{}: {}\n".format(key, self.__dict__[key])

        return return_str


def log_y_nd(log_p, n, label=False):

    if len(log_p.shape) == 2:
        if label is True:
            return np.r_[log_p[:n, :] - log_p[:1, :], log_p[n:, :] - log_p[:-n, :], log_p[-1:, :] - log_p[-n:, :]]
        else:
            return np.r_[log_p[:n, :] - log_p[:1, :], log_p[n:, :] - log_p[:-n, :]]
    elif len(log_p.shape) == 1:
        if label is True:
            return np.r_[log_p[:n] - log_p[:1], log_p[n:] - log_p[:-n], log_p[-1:] - log_p[-n:]]
        else:
            return np.r_[log_p[:n] - log_p[:1], log_p[n:] - log_p[:-n]]
    else:
        raise NotImplementedError


def strategy_based_label(log_p, n, cash_idx=0, base_weight=None):
    # log_p = idx_logp; n = 250
    y = np.exp(np.r_[np.zeros([1, log_p.shape[1]]), log_p[1:, :] - log_p[:-1, :]]) - 1.

    F_mat = np.zeros_like(y)
    for t in range(1, len(y)):
        m = np.nanmean(y[max(0, t-n+1):(t+1)], axis=0, keepdims=True)
        s = np.nanstd(y[max(0, t-n+1):(t+1)], axis=0, ddof=1, keepdims=True) + 1e-6

        # F = 1 + np.tanh(np.sqrt(250) * (m / s))
        # F = np.tanh(np.minimum(1, m / s ** 2)) + 1

        if base_weight is None:
            # kelly based label (equal)
            F = np.tanh(m / s ** 2) + 1
            nan_filter = np.isnan(F)
            n_asset = log_p.shape[1] - np.sum(nan_filter)
            F[nan_filter] = 0.
            F[~nan_filter] = np.maximum(1 / (3 * n_asset), F[~nan_filter])
            wgt = F / (n_asset * 2)
            wgt[:, cash_idx] = np.min([0.4, wgt[:, cash_idx] + 1 - np.sum(wgt[:])])
            wgt *= 1. / np.sum(wgt[:])
            F_mat[t, :] = wgt
            # F_mat[t, 0] = F_mat[t, 0] + np.max(0.5, 1 - np.sum(F / (n_asset * 2)))
        else:
            F_mat[t, :] = np.array(base_weight)
        # else:
        #     assert type(base_weight) == list
        #     assert len(base_weight) == log_p.shape[1]
        #     base_weight_arr = np.array(base_weight)
        #     max_base_weight = np.zeros_like(base_weight)
        #     min_base_weight = np.zeros_like(base_weight)
        #     cash_filter = np.zeros_like(base_weight, dtype=bool)
        #     cash_filter[cash_idx] = True
        #
        #     max_base_weight[~cash_filter] = base_weight_arr[~cash_filter] * (1+0.3)
        #     min_base_weight[~cash_filter] = base_weight_arr[~cash_filter] * (1-0.3)
        #
        #     # max_base_weight[cash_filter] = 1.
        #     # min_base_weight[cash_filter] = 1.
        #
        #     # kelly based label (equal)
        #     F = np.tanh(m / s ** 2) + 1
        #     nan_filter = np.isnan(F)
        #     n_asset = log_p.shape[1] - np.sum(nan_filter)
        #     F[nan_filter] = 0.
        #     F[~nan_filter] = np.minimum(max_base_weight, np.maximum(min_base_weight, F[~nan_filter]))
        #     F[:, cash_filter] = np.maximum(0, 1 - np.sum(F[:]))
        #
        #     F *= 1. / np.sum(F[:])
        #     F_mat[t, :] = F

    plot = False
    if plot:
        from matplotlib import pyplot as plt, cm
        viridis = cm.get_cmap('viridis', 8)
        x = np.arange(len(F_mat))
        wgt_base_cum = F_mat.cumsum(axis=1)
        fig = plt.figure()
        fig.suptitle('Weight Diff')
        ax1 = fig.add_subplot(311)
        ax1.set_title('base')
        for i in range(8):
            if i == 0:
                ax1.fill_between(x, 0, wgt_base_cum[:, i], facecolor=viridis.colors[i], alpha=.7)
            else:
                ax1.fill_between(x, wgt_base_cum[:, i - 1], wgt_base_cum[:, i], facecolor=viridis.colors[i], alpha=.7)

    return F_mat


def arr_to_normal_ts(arr):
    # time series normalize
    return_value = (arr - np.nanmean(arr, axis=0, keepdims=True)) / (np.nanstd(arr, axis=0, ddof=1, keepdims=True) + 1e-6)
    return return_value

@profile
def get_data(configs):
    # label days: target하는 n days 수익률
    # k days: 향후 k days 간의 수익률 (backtest용)

    # normalizing_window = 500; label_days = 60; k_days = 20; test_days = 240
    # parameters
    c = configs

    calc_days = [20, 60, 120, 250]  # list of return calculation days
    # normalizing_window = 500  # macro normalizing window size
    # label_days = 60
    # k_days = 20
    # test_days = 240

    delete_nan = 500

    # get data
    macro_data = pd.read_csv('./data/macro_data.txt', index_col=0, sep='\t')
    macro_data['copper_gold_r'] = macro_data['HG1 Comdty'] / macro_data['GC1 Comdty']

    if c.datatype == 'app':
        idx_data = pd.read_csv('./data/app_data.txt', index_col=0, sep='\t')
    else:
        idx_data = pd.read_csv('./data/index_data.txt', index_col=0, sep='\t')

    print(idx_data.columns)
    merged = pd.merge(macro_data, idx_data, left_index=True, right_index=True)
    macro = merged.to_numpy()[:, :len(macro_data.columns)]
    idx = merged.to_numpy()[:, len(macro_data.columns):]

    # macro features
    n_point, n_macro = macro.shape
    macro_features = np.zeros([n_point, n_macro])
    for t in range(c.normalizing_window, n_point):
        macro_features[t] = arr_to_normal_ts(macro[(t-c.normalizing_window+1):(t+1)])[-1]

    # idx features
    idx_logp = np.log(idx)
    n_point, n_asset = idx_logp.shape

    idx_features = np.zeros([n_point, n_asset * len(calc_days)])
    wgt_features = np.zeros([n_point, n_asset])

    i = 0
    for i_days in calc_days:
        idx_features[:, i:(i+n_asset)] = log_y_nd(idx_logp, i_days, label=False)
        i += n_asset

    wgt_features[:, :] = strategy_based_label(idx_logp, c.strategy_days, cash_idx=c.cash_idx, base_weight=c.base_weight)

    # labels
    labels_dict = dict()
    # 1은 매매 실현가능성 위해
    labels_dict['logy'] = log_y_nd(idx_logp, c.label_days, label=True)[(c.k_days+1):][delete_nan:]
    labels_dict['wgt'] = wgt_features[(c.k_days + 1):][delete_nan:]
    labels_dict['wgt'] = np.r_[labels_dict['wgt'], np.repeat(labels_dict['wgt'][-1:], c.k_days, axis=0)]
    # labels_dict['wgt'] = strategy_based_label(idx_logp, label_days)[(label_days+1):][delete_nan:]
    labels_dict['logy_for_calc'] = log_y_nd(idx_logp, c.k_days, label=True)[(c.k_days+1):][delete_nan:]

    # features and labels
    features_dict = dict()
    features_dict['macro'] = macro_features[delete_nan:]
    features_dict['idx'] = idx_features[delete_nan:]
    features_dict['wgt'] = wgt_features[delete_nan:]

    add_info = dict()
    add_info['date'] = merged.index.to_numpy()[delete_nan:]
    add_info['idx_list'] = idx_data.columns.to_numpy()
    add_info['macro_list'] = macro_data.columns.to_numpy()
    add_info['calc_days'] = calc_days

    # truncate unlabeled data
    label_len = np.min([len(labels_dict['logy']), len(labels_dict['wgt']), len(labels_dict['logy_for_calc'])])
    for key in labels_dict.keys():
        labels_dict[key] = labels_dict[key][:label_len]

    for key in features_dict.keys():
        features_dict[key] = features_dict[key][:label_len]

    add_info['date'] = add_info['date'][:label_len]

    return features_dict, labels_dict, add_info  # labels가 features 보다 label_days만큼 짧음


def to_torch(dataset):
    features_dict, labels_dict = dataset
    for key in features_dict.keys():
        features_dict[key] = torch.tensor(features_dict[key], dtype=torch.float32)

    for key in labels_dict.keys():
        labels_dict[key] = torch.tensor(labels_dict[key], dtype=torch.float32)

    return [features_dict, labels_dict]


def normalize_action(action):
    action = torch.sigmoid(action) + 1e-3
    action = action / action.sum(dim=1, keepdim=True)
    return action


def main():

    name = 'gcsl_test_05'
    c = Configs(name)

    str_ = c.export()
    with open(os.path.join(c.outpath, 'c.txt'), 'w') as f:
        f.write(str_)

    # data processing
    features_dict, labels_dict, add_info = get_data(configs=c)
    sampler = Sampler(features_dict, labels_dict, add_info, configs=c)

    model = MyModel(sampler.n_features + 4, sampler.n_labels, cost_rate=c.cost_rate) # 4: nav, mdd, g, h
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=0.01)

    memory = Memory(1e5)

    t = 3001
    outpath_t = os.path.join(c.outpath, str(t))
    os.makedirs(outpath_t, exist_ok=True)
    # initial data collecting
    for _ in range(1000):
        goal = 0.03 + np.random.rand() * 0.07   # 0.03~0.1
        idx = np.random.choice(np.arange(500, t))
        mem_per_traj, _ = sampler.sample_trajectory(goal, model, sampler.sample_env(idx, data_len=2500))
        memory.add(mem_per_traj)

    model.train()
    batch_size = 256

    for ep in range(1000):
        if ep % 10 == 0:
            # plot
            _, [traj, pf_r, pf_mdd] = sampler.sample_trajectory(0.07, model, sampler.sample_env(t, data_len=2500))
            a_df = pd.DataFrame(columns=[0, 1, 2, 3])
            for i_tr, tr in enumerate(traj):
                a_df.loc[i_tr] = list(tu.np_ify(tr[1]).squeeze())

            data_for_plot = pd.DataFrame({'r': tu.np_ify(pf_r.squeeze())[:len(traj)], 'mdd': 1 + tu.np_ify(pf_mdd.squeeze())[:len(traj)]}, index=np.arange(len(pf_r))[:len(traj)])
            fig, axes = plt.subplots(nrows=2, ncols=1)
            # ax1 = fig.add_subplot(211)
            axes[0].plot(data_for_plot)
            a_df.plot.area(ax=axes[1])
            fig.savefig(os.path.join(outpath_t, 'test_{}.png'.format(ep)))
            plt.close(fig)

        # new trj.
        goal = 0.03 + np.random.rand() * 0.07   # 0.03~0.1
        idx = np.random.choice(np.arange(500, t))
        mem_per_traj, _ = sampler.sample_trajectory(goal, model, sampler.sample_env(idx, data_len=2500))
        memory.add(mem_per_traj)


        ep_losses = 0
        n_batch_cycle = memory.reset_epoch(batch_size)
        for _ in range(n_batch_cycle):
            data = memory.sample_batch(batch_size, epoch=True)
            # pred_a = model.policy(data['s0'], data['sh'], data['h'])
            policy_dist, mu, scale = model.policy(data['s0'], data['sh'], data['h'])
            pred_a = normalize_action(policy_dist.rsample())
            label_a = data['a']
            entropy = policy_dist.entropy()
            # loss = entropy.sum()
            # loss = (nn.MSELoss(reduction='none')(pred_a, label_a) * data['sign']).sum() - entropy.sum()
            loss = (nn.KLDivLoss(reduction='none')(torch.log(pred_a), label_a) * data['sign']).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_losses += tu.np_ify(loss)

        print(ep, ep_losses)


class Memory(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory_counter = 0
        self.memory = list()

    def add(self, memory_per_traj):
        self.memory += memory_per_traj

        excess = int(len(self.memory) - self.memory_size)
        if excess > 0:
            self.memory = self.memory[excess:]

        self.memory_counter = len(self.memory)

    def clear(self):
        self.memory = list()
        self.memory_counter = 0

    def sample_batch(self, batch_size, epoch=True):
        if self.memory_counter < batch_size:
            # print('insufficient memory')
            sampled_memory = random.sample(self.memory, self.memory_counter)
            # return False
        else:
            if epoch is True:
                if len(self.epoch_remain_idx) < batch_size:
                    sampled_memory = [self.memory[i] for i in self.epoch_remain_idx]
                else:
                    sampled_idx = self.epoch_remain_idx[:batch_size]
                    self.epoch_remain_idx = self.epoch_remain_idx[batch_size:]
                    sampled_memory = [self.memory[i] for i in sampled_idx]
            else:
                sampled_memory = random.sample(self.memory, batch_size)

        sampled_batch = dict(s0=list(), a=list(), sh=list(), h=list(), sign=list())
        for key in sampled_batch.keys():
            for i in range(len(sampled_memory)):
                sampled_batch[key].append(sampled_memory[i][key])
            sampled_batch[key] = torch.cat(sampled_batch[key], dim=0)

        return sampled_batch

    def reset_epoch(self, batch_size):
        self.epoch_remain_idx = np.arange(len(self.memory))
        np.random.shuffle(self.epoch_remain_idx)
        self.epoch_remain_idx = list(self.epoch_remain_idx)
        return int(np.ceil(len(self.epoch_remain_idx) / batch_size))


    # sampler = Sampler(features, labels, init_train_len=500, label_days=130)
# train_dataset, eval_dataset, test_dataset = sampler.get_batch(1000)

class Sampler:
    def __init__(self, features, labels, add_infos, configs):
        c = configs
        self.features = features
        self.labels = labels
        self.add_infos = add_infos
        self.init_train_len = c.init_train_len
        self.label_days = c.label_days
        self.test_days = c.test_days
        self.use_accum_data = c.use_accum_data

        self.n_features = len(add_infos['macro_list']) + len(add_infos['idx_list']) * len(add_infos['calc_days'])
        self.n_labels = len(add_infos['idx_list'])

    @property
    def date_(self):
        return self.add_infos['date']

    @property
    def max_len(self):
        return len(self.add_infos['date']) - 1

    def sample_trajectory(self, goal, model, dataset):
        # dataset =  sampler.sample_env(idx); self = sampler
        npoint_per_year = 250 // self.label_days

        traj = list()
        memory_per_traj = list()
        features, labels = to_torch(dataset)
        x = torch.cat([features['idx'], features['macro']], dim=-1)
        max_h = len(x) // self.label_days

        pf_r = torch.ones(len(features['idx']) // self.label_days + 1, 1)
        pf_mdd = torch.zeros_like(pf_r)
        r = 0
        h = max_h / 100
        for i, t in enumerate(range(0, len(x)-1, self.label_days)):
            with torch.set_grad_enabled(False):
                s0 = torch.cat([x[t:(t+1)], pf_r[i:(i+1)], pf_mdd[i:(i+1)]], dim=1)
                policy, mu, scale = model.policy(s0, goal, h)
                a = normalize_action(mu)
                h -= 1 / 100
                traj.append([s0, a])

                pf_r[i+1] = pf_r[i] * (1. + (a * (torch.exp(labels['logy_for_calc'][t])-1.)).sum())
                pf_mdd[i+1] = pf_r[i+1] / pf_r[:(i+2)].max() - 1.

            if pf_mdd[i+1] <= -0.03:
                r = -1
                break

            if i % npoint_per_year == 0 and pf_r[-1] < (1 + goal) ** (i//npoint_per_year):
                r = -1
                break

        if r == 0 and pf_r[-1] < (1+goal) ** (i/npoint_per_year):
            r = -1
        else:
            r = 1

        for k in range(1, len(traj)):
            if k == (len(traj)-1) and r == -1:
            # if r == -1:
                sign_ = -1
            else:
                sign_ = 1
            for m in range(k):
                memory_per_traj.append({'s0': traj[m][0],
                                        'a':  traj[m][1],
                                        'sh': pf_r[k:(k+1)],
                                        'h': torch.tensor([[(k-m) / 100]], dtype=torch.float32),
                                        'sign': torch.tensor([[sign_]], dtype=torch.float32)})

        return memory_per_traj, [traj, pf_r, pf_mdd]

    def sample_env(self, i, data_len=250):
        assert i >= self.init_train_len

        start_i = i - data_len
        end_i = i
        idx = np.arange(start_i, end_i)
        features_train = dict([(key, self.features[key][idx]) for key in self.features.keys()])
        labels_train = dict([(key, self.labels[key][idx]) for key in self.labels.keys()])
        dataset = (features_train, labels_train)

        return dataset

    def get_batch(self, i):
        train_data_len = 1000
        assert i >= self.init_train_len

        if self.use_accum_data:
            train_base_i = 0
            eval_base_i = int(train_base_i + 0.6 * (i - train_base_i))
        else:
            train_base_i = max(0, i - train_data_len)
            eval_base_i = int(train_base_i + 0.6 * min(train_data_len, i - train_base_i))

        test_base_i = i

        train_start_i = train_base_i
        train_end_i = eval_base_i - self.label_days
        eval_start_i = eval_base_i
        eval_end_i = test_base_i - self.label_days
        test_start_i = test_base_i
        test_end_i = min(i + self.test_days, self.max_len)

        # train_idx = np.random.choice(np.arange(train_start_i, train_end_i), train_end_i-train_start_i, replace=False)
        train_idx = np.arange(train_start_i, train_end_i)
        features_train = dict([(key, self.features[key][train_idx]) for key in self.features.keys()])
        labels_train = dict([(key, self.labels[key][train_idx]) for key in self.labels.keys()])
        train_dataset = (features_train, labels_train)
        # train_dataset = (self.features[train_idx], self.labels[train_idx], self.sr_labels[train_idx])
        eval_idx = np.random.choice(np.arange(eval_start_i, eval_end_i), eval_end_i-eval_start_i, replace=False)
        features_eval = dict([(key, self.features[key][eval_idx]) for key in self.features.keys()])
        labels_eval = dict([(key, self.labels[key][eval_idx]) for key in self.labels.keys()])
        eval_dataset = (features_eval, labels_eval)
        # eval_dataset = (self.features[eval_idx], self.labels[eval_idx], self.sr_labels[eval_idx])
        # test_idx = np.random.choice(np.arange(test_start_i, test_end_i), test_end_i-test_start_i, replace=False)
        test_idx = np.arange(test_start_i, test_end_i)
        features_test = dict([(key, self.features[key][test_idx]) for key in self.features.keys()])
        labels_test = dict([(key, self.labels[key][test_idx]) for key in self.labels.keys()])
        test_dataset = (features_test, labels_test)
        # test_dataset = (self.features[test_idx], self.labels[test_idx], self.sr_labels[test_idx])

        test_insample_idx = np.arange(train_start_i, test_end_i)
        features_test_insample = dict([(key, self.features[key][test_insample_idx]) for key in self.features.keys()])
        labels_test_insample = dict([(key, self.labels[key][test_insample_idx]) for key in self.labels.keys()])
        test_dataset_insample = (features_test_insample, labels_test_insample)
        insample_boundary = np.concatenate([np.where(test_insample_idx == eval_start_i)[0], np.where(test_insample_idx == test_start_i)[0]])
        guide_date = [self.date_[train_start_i], self.date_[eval_start_i], self.date_[test_start_i], self.date_[test_end_i]]
        return train_dataset, eval_dataset, test_dataset, (test_dataset_insample, insample_boundary), guide_date


class XLinear(Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(XLinear, self).__init__()
        self.layer = Linear(in_dim, out_dim, bias)
        init.xavier_uniform_(self.layer.weight)
        if bias:
            init.zeros_(self.layer.bias)

    def forward(self, x):
        return self.layer(x)


class MyModel(Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[72, 48, 32], dropout_r=0.5, cost_rate=0.003):
        super(MyModel, self).__init__()
        self.cost_rate = cost_rate
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout_r = dropout_r
        self.hidden_layers = nn.ModuleList()

        h_in = in_dim
        for h_out in hidden_dim:
            self.hidden_layers.append(XLinear(h_in, h_out))
            h_in = h_out

        self.mu_out_layer = XLinear(h_out, out_dim)
        self.logscale_out_layer = XLinear(h_out, out_dim)

        self.loss_func_logy = nn.MSELoss()

        self.optim_state_dict = self.state_dict()
        # self.optim_
    # def init_weight(self):
    #     return torch.ones_like()

    def forward(self, x, sample=True):
        mask = self.training or sample

        for h_layer in self.hidden_layers:
            x = h_layer(x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout_r, training=mask)

        mu = torch.sigmoid(self.mu_out_layer(x))
        scale = self.logscale_out_layer(x)
        scale = torch.clamp(scale, 0.01, 0.5)
        # logscale = self.logscale_out_layer(x)
        # logscale = torch.clamp(logscale, -20., 2.)
        # scale = logscale.exp()

        policy_dist = torch.distributions.Normal(mu, torch.exp(scale))
        # entropy = policy_dist.entropy()
        return policy_dist, mu, scale

    def adversarial_noise(self, features, labels):
        for key in features.keys():
            features[key].requires_grad = True

        for key in labels.keys():
            labels[key].requires_grad = True

        pred, losses, _, _, _ = self.forward_with_loss(features, labels, n_samples=100, loss_wgt=None)

        losses.backward(retain_graph=True)
        features_grad = dict()
        features_perturbed = dict()
        for key in features.keys():
            features_grad[key] = torch.sign(features[key].grad)
            sample_sigma = torch.std(features[key], axis=[0], keepdims=True)
            eps = 0.01
            scaled_eps = eps * sample_sigma  # [1, 1, n_features]

            features_perturbed[key] = features[key] + scaled_eps * features_grad[key]
            features[key].grad.zero_()

        return features_perturbed

    @profile
    def policy(self, s, g, h):
        # features, labels=  train_features, train_labels
        # s = s0; self = model; n_samples=10
        if type(g) in [int, float]:
            g = torch.tensor([[g]], dtype=torch.float32).to(s.device)

        if type(h) in [int, float]:
            h = torch.tensor([[h]], dtype=torch.float32).to(s.device)

        x = torch.cat([s, g, h], dim=-1)
        x = self.forward(x)

        return x

    def save_to_optim(self):
        self.optim_state_dict = self.state_dict()

    def load_from_optim(self):
        self.load_state_dict(self.optim_state_dict)

