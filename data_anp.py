import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

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

        # F = 1 + np.tanh(np.sqrt(250) * (app_m.yaml / s))
        # F = np.tanh(np.minimum(1, app_m.yaml / s ** 2)) + 1

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
        #     F = np.tanh(app_m.yaml / s ** 2) + 1
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

    calc_days = [20,]  # list of return calculation days
    # normalizing_window = 500  # macro normalizing window size
    # label_days = 60
    # k_days = 20
    # test_days = 240

    delete_nan = 500

    # get data_conf
    macro_data = pd.read_csv('./data_conf/macro_data.txt', index_col=0, sep='\t')
    macro_data['copper_gold_r'] = macro_data['HG1 Comdty'] / macro_data['GC1 Comdty']

    if c.datatype == 'app':
        idx_data = pd.read_csv('./data_conf/app_data.txt', index_col=0, sep='\t')
    else:
        idx_data = pd.read_csv('./data_conf/index_data.txt', index_col=0, sep='\t')

    min_begin_i = np.max(np.isnan(idx_data).sum(axis=0)) + 1

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
    add_info['min_begin_i'] = min_begin_i

    # truncate unlabeled data_conf
    label_len = np.min([len(labels_dict['logy']), len(labels_dict['wgt']), len(labels_dict['logy_for_calc'])])
    for key in labels_dict.keys():
        labels_dict[key] = labels_dict[key][:label_len]

    for key in features_dict.keys():
        features_dict[key] = features_dict[key][:label_len]

    add_info['date'] = add_info['date'][:label_len]

    return features_dict, labels_dict, add_info  # labels가 features 보다 label_days만큼 짧음


def to_torch(dataset):
    features_prev_dict, features_dict, labels_dict = dataset
    for key in features_dict.keys():
        features_prev_dict[key] = torch.tensor(features_prev_dict[key], dtype=torch.float32)
        features_dict[key] = torch.tensor(features_dict[key], dtype=torch.float32)

    for key in labels_dict.keys():
        labels_dict[key] = torch.tensor(labels_dict[key], dtype=torch.float32)

    return [features_prev_dict, features_dict, labels_dict]


# sampler = Sampler(features, labels, init_train_len=500, label_days=130)
# train_dataset, eval_dataset, test_dataset = sampler.get_batch(1000)
class Sampler:
    def __init__(self, features, labels, add_infos, configs):
        c = configs
        self.features = features
        self.labels = labels
        self.add_infos = add_infos
        self.init_train_len = c.init_train_len

        self.train_data_len = c.train_data_len
        self.label_days = c.label_days
        self.sampling_freq = c.sampling_freq
        self.test_days = c.test_days
        self.use_accum_data = c.use_accum_data

        self.n_features = len(add_infos['macro_list']) + len(add_infos['idx_list']) * len(add_infos['calc_days']) + 1
        self.n_labels = len(add_infos['idx_list'])

    @property
    def date_(self):
        return self.add_infos['date']

    @property
    def max_len(self):
        return len(self.add_infos['date']) - 1

    def get_batch(self, i):
        assert i >= self.init_train_len

        if self.use_accum_data:
            train_base_i = self.add_infos['min_begin_i']
            eval_base_i = int(train_base_i + 0.6 * (i - train_base_i))
        else:
            train_base_i = max(self.add_infos['min_begin_i'], i - self.train_data_len)
            eval_base_i = int(train_base_i + 0.6 * min(self.train_data_len, i - train_base_i))

        test_base_i = i

        train_start_i = train_base_i
        train_end_i = eval_base_i - self.label_days
        eval_start_i = eval_base_i
        eval_end_i = test_base_i - self.label_days
        test_start_i = test_base_i
        test_end_i = min(i + self.test_days, self.max_len)

        train_idx = np.random.choice(np.arange(train_start_i + 1, train_end_i), train_end_i-train_start_i-1, replace=False)
        features_train_prev = dict([(key, self.features[key][train_idx-1]) for key in self.features.keys()])
        features_train = dict([(key, self.features[key][train_idx]) for key in self.features.keys()])
        labels_train = dict([(key, self.labels[key][train_idx]) for key in self.labels.keys()])
        train_dataset = (features_train_prev, features_train, labels_train)
        # train_dataset = (self.features[train_idx], self.labels[train_idx], self.sr_labels[train_idx])
        eval_idx = np.random.choice(np.arange(eval_start_i + 1, eval_end_i), eval_end_i-eval_start_i-1, replace=False)
        features_eval_prev = dict([(key, self.features[key][eval_idx-1]) for key in self.features.keys()])
        features_eval = dict([(key, self.features[key][eval_idx]) for key in self.features.keys()])
        labels_eval = dict([(key, self.labels[key][eval_idx]) for key in self.labels.keys()])
        eval_dataset = (features_eval_prev, features_eval, labels_eval)
        # eval_dataset = (self.features[eval_idx], self.labels[eval_idx], self.sr_labels[eval_idx])
        # test_idx = np.random.choice(np.arange(test_start_i, test_end_i), test_end_i-test_start_i, replace=False)
        test_idx = np.arange(test_start_i + 1, test_end_i)
        features_test_prev = dict([(key, self.features[key][test_idx-1]) for key in self.features.keys()])
        features_test = dict([(key, self.features[key][test_idx]) for key in self.features.keys()])
        labels_test = dict([(key, self.labels[key][test_idx]) for key in self.labels.keys()])
        test_dataset = (features_test_prev, features_test, labels_test)
        # test_dataset = (self.features[test_idx], self.labels[test_idx], self.sr_labels[test_idx])

        test_insample_idx = np.arange(train_start_i + 1, test_end_i)
        features_test_insample_prev = dict([(key, self.features[key][test_insample_idx-1]) for key in self.features.keys()])
        features_test_insample = dict([(key, self.features[key][test_insample_idx]) for key in self.features.keys()])
        labels_test_insample = dict([(key, self.labels[key][test_insample_idx]) for key in self.labels.keys()])
        test_dataset_insample = (features_test_insample_prev, features_test_insample, labels_test_insample)
        insample_boundary = np.concatenate([np.where(test_insample_idx == eval_start_i)[0], np.where(test_insample_idx == test_start_i)[0]])
        guide_date = [self.date_[train_start_i], self.date_[eval_start_i], self.date_[test_start_i], self.date_[test_end_i]]
        return train_dataset, eval_dataset, test_dataset, (test_dataset_insample, insample_boundary), guide_date

    def get_batch_set(self, i, is_train=True):
        assert i >= self.init_train_len

        batch_set = []

        context_len = 24

        start_i = max(self.add_infos['min_begin_i'],  context_len * self.sampling_freq)
        end_i = i - self.label_days

        total_idx = np.arange(start_i + 1, end_i)

        test_start_i = i
        test_end_i = min(i + self.test_days, self.max_len)

        t_idx = ((np.arange(context_len + 1) - context_len) / 10).reshape([-1, 1])
        for t in total_idx:
            selected_i = np.arange(t - context_len*self.sampling_freq, t+1, self.sampling_freq)
            shuffled_i = np.arange(len(selected_i[:-1])).ravel()
            np.random.shuffle(shuffled_i)

            context_i = shuffled_i[:int(len(shuffled_i) * 0.7)]
            target_i = np.append(shuffled_i, len(shuffled_i))
            if not is_train:
                target_i = np.sort(target_i)

            context_x = np.concatenate([self.features['macro'][selected_i[context_i]],
                                        self.features['idx'][selected_i[context_i]],
                                        t_idx[context_i]], axis=1)

            context_y = self.labels['logy_for_calc'][selected_i[context_i]]

            target_x = np.concatenate([self.features['macro'][selected_i[target_i]],
                                       self.features['idx'][selected_i[target_i]],
                                       t_idx[target_i]], axis=1)
            target_y = self.labels['logy_for_calc'][selected_i[target_i]]

            batch_set.append([context_x, context_y, target_x, target_y])
            # batch_set.append([torch.from_numpy(context_x).float().to(tu.device), torch.from_numpy(context_y).float().to(tu.device),
            #                   torch.from_numpy(target_x).float().to(tu.device), torch.from_numpy(target_y).float().to(tu.device)])

        return batch_set








