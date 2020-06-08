from copy import deepcopy
import random
import re
import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt, cm


from model import MyModel, load_model, save_model
from data import get_data, Sampler, to_torch, data_loader
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
        self.lr = 5e-4
        self.batch_size = 512
        self.num_epochs = 2000
        self.base_i0 = 2000
        self.mc_samples = 200
        self.sampling_freq = 20
        self.k_days = 20
        self.label_days = 20
        self.strategy_days = 250
        self.adaptive_count = 5
        self.adaptive_lrx = 2 # learning rate * 배수

        self.es_type = 'train'  # 'eval'
        self.es_max_count = 20
        self.retrain_days = 240
        self.test_days = 2000  # test days
        self.init_train_len = 500
        self.train_data_len = 2000
        self.normalizing_window = 500  # norm windows for macro data
        self.use_accum_data = True # [sampler] 데이터 누적할지 말지
        self.adaptive_flag = True
        self.adv_train = True
        self.n_pretrain = 5


        self.loss_threshold = None # -1
        self.adaptive_loss_threshold = None # -1

        self.datatype = 'app'
        # self.datatype = 'inv'

        self.cost_rate = 0.003
        self.plot_freq = 10
        self.eval_freq = 1 # 20
        self.save_freq = 20
        self.model_init_everytime = True

        self.hidden_dim = [72, 48, 32]
        self.dropout_r = 0.3

        self.random_guide_weight = 0.0
        self.random_label = 0.1  # flip sign

        self.clip = 1.

        self.loss_wgt = {'y_pf': 1., 'mdd_pf': 1., 'logy': 1., 'wgt': 0., 'wgt2': 1., 'wgt_guide': 0., 'cost': 0., 'entropy': 0.}
        self.adaptive_loss_wgt = {'y_pf': 1, 'mdd_pf': 1000., 'logy': 1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.02, 'cost': 1., 'entropy': 0.001}

        # good balance
        # self.adaptive_loss_wgt = {'y_pf': 1, 'mdd_pf': 1000., 'logy': 1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.0002,
        #                       'cost': 2., 'entropy': 0.0001}
        self.init()
        self.set_path()

    def init(self):
        self.init_weight()
        self.init_loop()

    def init_weight(self):
        if self.datatype == 'app':
            self.cash_idx = 3
            # self.base_weight = [0.7, 0.2, 0.1, 0.0]
            # self.base_weight = [0.4, 0.15, 0.075, 0.375]
            self.base_weight = [0.25, 0.1, 0.05, 0.6]
            # self.base_weight = [0.4, 0.3, 0.3, 0.0]
            # self.base_weight = None
        else:
            self.cash_idx = 0
            self.base_weight = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
            # self.base_weight = None

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
        # self.adaptive_loss_wgt = {'y_pf': 0.2, 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.05,
        #                           'cost': 1., 'entropy': 0.001}

        return adaptive_flag

    def export(self):
        return_str = ""
        for key in self.__dict__.keys():
            return_str += "{}: {}\n".format(key, self.__dict__[key])

        return return_str


def calc_y(wgt0, y1, cost_r=0.):
    # wgt0: 0 ~ T-1 ,  y1 : 1 ~ T  => 0 ~ T (0번째 값은 0)
    y = dict()
    wgt1 = wgt0 * (1 + y1)
    turnover = np.append(np.sum(np.abs(wgt0[1:] - wgt1[:-1]), axis=1), 0)
    y['before_cost'] = np.insert(np.sum(wgt1, axis=1) - 1, 0, 0)
    y['after_cost'] = np.insert(np.sum(wgt1, axis=1) - 1 - turnover * cost_r, 0, 0)

    return y, turnover


def plot_each(ep, sampler, model, features, labels, insample_boundary=None, guide_date=None, mc_samples=100, k_days=20, cost_rate=0.003, suffix='', outpath='./out/', guide_weight=None):
    """

    features, labels =  f_dict['test_insample'], l_dict['test_insample']
    mc_samples=c.mc_samples
    k_days=c.k_days
    outpath=outpath_t
    cost_rate=c.cost_rate
    guide_weight=c.base_weight
    """

    # ep=0; n_samples = 100; k_days = 20; model, features, labels, insample_boundary = main(testmode=True)
    # features, labels = test_features, test_labels
    # features, labels = test_insample_features, test_insamples_labels

    with torch.set_grad_enabled(False):
        wgt_test, losses_test, pred_mu_test, pred_sigma_test, _ = model.forward_with_loss(features, None, mc_samples=mc_samples, is_train=False)

    wgt_base = tu.np_ify(features['wgt'])
    wgt_label = tu.np_ify(labels['wgt'])
    wgt_result = tu.np_ify(wgt_test)
    n_asset = wgt_base.shape[1]

    viridis = cm.get_cmap('viridis', n_asset)

    # weight change
    x = np.arange(len(wgt_base))
    wgt_base_cum = wgt_base.cumsum(axis=1)
    fig = plt.figure()
    fig.suptitle('Weight Diff')
    # ax1 = fig.add_subplot(311)
    ax1 = fig.add_subplot(211)
    ax1.set_title('base')
    for i in range(n_asset):
        if i == 0:
            ax1.fill_between(x, 0, wgt_base_cum[:, i], facecolor=viridis.colors[i], alpha=.7)
        else:
            ax1.fill_between(x, wgt_base_cum[:, i-1], wgt_base_cum[:, i], facecolor=viridis.colors[i], alpha=.7)

    wgt_result_cum = wgt_result.cumsum(axis=1)
    # ax2 = fig.add_subplot(312)
    ax2 = fig.add_subplot(212)
    ax2.set_title('result')
    for i in range(n_asset):
        if i == 0:
            ax2.fill_between(x, 0, wgt_result_cum[:, i], facecolor=viridis.colors[i], alpha=.7)
        else:
            ax2.fill_between(x, wgt_result_cum[:, i-1], wgt_result_cum[:, i], facecolor=viridis.colors[i], alpha=.7)

    # wgt_label_cum = wgt_label.cumsum(axis=1)
    # ax3 = fig.add_subplot(313)
    # ax1 = fig.add_subplot(313)
    # ax3.set_title('label')
    # for i in range(n_asset):
    #     if i == 0:
    #         ax3.fill_between(x, 0, wgt_label_cum[:, i], facecolor=viridis.colors[i], alpha=.7)
    #     else:
    #         ax3.fill_between(x, wgt_label_cum[:, i-1], wgt_label_cum[:, i], facecolor=viridis.colors[i], alpha=.7)

    if insample_boundary is not None:
        ax1.axvline(insample_boundary[0])
        ax1.axvline(insample_boundary[1])

        ax2.axvline(insample_boundary[0])
        ax2.axvline(insample_boundary[1])

        # ax3.axvline(insample_boundary[0])
        # ax3.axvline(insample_boundary[1])

        ax1.text(x[0], ax1.get_ylim()[1], guide_date[0]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))
        ax1.text(insample_boundary[0], ax1.get_ylim()[1], guide_date[1]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))
        ax2.text(insample_boundary[1], ax1.get_ylim()[1], guide_date[2]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))
        ax2.text(x[-1], ax1.get_ylim()[1], guide_date[3]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))

    fig.savefig(os.path.join(outpath, 'test_wgt_{}{}.png'.format(ep, suffix)))
    plt.close(fig)

    # #######################
    y_next = tu.np_ify(torch.exp(labels['logy_for_calc'])-1.)[::k_days, :]
    wgt_base_calc = wgt_base[::k_days, :]
    wgt_result_calc = wgt_result[::k_days, :]
    wgt_label_calc = wgt_label[::k_days, :]

    # features, labels = test_features, test_labels; wgt = wgt_result
    # active_share = np.sum(np.abs(wgt_result - wgt_base), axis=1)
    active_share = np.sum(np.abs(wgt_result_calc - wgt_base_calc), axis=1)

    y_base = np.insert(np.sum((1+y_next) * wgt_base_calc, axis=1), 0, 1.)
    y_port = np.insert(np.sum((1+y_next) * wgt_result_calc, axis=1), 0, 1.)
    y_label = np.insert(np.sum((1+y_next) * wgt_label_calc, axis=1), 0, 1.)
    y_eq = np.insert(np.mean(1+y_next, axis=1), 0, 1.)



    if guide_weight is not None:
        y_guide = np.insert(np.sum((1+y_next) * np.array(guide_weight), axis=1), 0, 1.)
    else:
        y_guide = y_eq

    x = np.arange(len(y_base))


    # save data
    if guide_date is not None:
        date_ = sampler.date_[(sampler.date_ >= guide_date[0]) & (sampler.date_ < guide_date[-1])]
        date_test_selected = ((date_ >= guide_date[-2]) & (date_ < guide_date[-1]))[::k_days]

        date_selected = list(date_)[::k_days]
        # date_base.append(sampler.add_infos['date'][t])

        y_base_with_c, turnover_base = calc_y(wgt_base_calc, y_next, cost_rate)
        y_port_with_c, turnover_port = calc_y(wgt_result_calc, y_next, cost_rate)
        y_label_with_c, turnover_label = calc_y(wgt_label_calc, y_next, cost_rate)

        idx_list = sampler.add_infos['idx_list']
        columns = [idx_nm + '_wgt' for idx_nm in idx_list] + [idx_nm + '_ynext' for idx_nm in idx_list] + ['port_bc', 'guide_bc', 'port_ac', 'guide_ac']
        df = pd.DataFrame(data=np.concatenate([wgt_result_calc, y_next,
                        y_port_with_c['before_cost'][1:, np.newaxis], y_label_with_c['before_cost'][1:, np.newaxis],
                        y_port_with_c['after_cost'][1:, np.newaxis], y_label_with_c['after_cost'][1:, np.newaxis]
                        ], axis=-1),
                          index=date_selected
                          , columns=columns)

        df_all = df.loc[:, ['port_bc', 'guide_bc', 'port_ac', 'guide_ac']]
        df_test = df.loc[date_test_selected, ['port_bc', 'guide_bc', 'port_ac', 'guide_ac']]
        df_stats = pd.concat({'mu_all': df_all.mean() * 12,
                           'sig_all': df_all.std(ddof=1) * np.sqrt(12),
                           'sr_all': df_all.mean() / df_all.std(ddof=1) * np.sqrt(12),
                           'mu_test': df_test.mean() * 12,
                           'sig_test': df_test.std(ddof=1) * np.sqrt(12),
                           'sr_test': df_test.mean() / df_test.std(ddof=1) * np.sqrt(12)},
                          axis=1)

        print(df_stats)
        df.to_csv(os.path.join(outpath, 'all_data.csv'))
        df_stats.to_csv(os.path.join(outpath, 'stats.csv'))

    # ################ together

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    # ax = plt.gca()
    l_port, = ax1.plot(x, y_port.cumprod())
    l_eq, = ax1.plot(x, y_eq.cumprod())
    l_guide, = ax1.plot(x, y_guide.cumprod())
    ax1.legend(handles=(l_port, l_eq, l_guide), labels=('port', 'eq', 'guide'))

    ax2 = fig.add_subplot(212)
    l_port_guide, = ax2.plot(x, (1 + y_port - y_guide).cumprod() - 1.)
    l_port_eq, = ax2.plot(x,(1 + y_port - y_eq).cumprod() - 1.)
    ax2.legend(handles=(l_port_guide, l_port_eq), labels=('port-guide', 'port-eq'))

    if insample_boundary is not None:
        ax1.axvline(insample_boundary[0] / k_days)
        ax1.axvline(insample_boundary[1] / k_days)
        ax2.axvline(insample_boundary[0] / k_days)
        ax2.axvline(insample_boundary[1] / k_days)

        ax1.text(x[0], ax1.get_ylim()[1], guide_date[0]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))
        ax1.text(insample_boundary[0] // k_days, ax1.get_ylim()[1], guide_date[1]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))
        ax2.text(insample_boundary[1] // k_days, ax2.get_ylim()[1], guide_date[2]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))
        ax2.text(x[-1], ax2.get_ylim()[1], guide_date[3]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))

    fig.savefig(os.path.join(outpath, 'test_y_{}{}.png'.format(ep, suffix)))
    plt.close(fig)


    # # ##################### seperate
    # fig = plt.figure()
    # ax = plt.gca()
    # # l_base, = plt.plot(x, y_base.cumprod())
    # l_port, = plt.plot(x, y_port.cumprod())
    # # l_label, = plt.plot(x, y_label.cumprod())
    # l_eq, = plt.plot(x, y_eq.cumprod())
    # l_guide, = plt.plot(x, y_guide.cumprod())
    # plt.legend(handles=(l_port, l_eq, l_guide), labels=('port', 'eq', 'guide'))
    # # plt.legend(handles=(l_base, l_port, l_label, l_eq, l_guide), labels=('base', 'port', 'label', 'eq', 'guide'))
    # if insample_boundary is not None:
    #     plt.axvline(insample_boundary[0] / k_days)
    #     plt.axvline(insample_boundary[1] / k_days)
    #
    #     plt.text(x[0], ax.get_ylim()[1], guide_date[0]
    #              , horizontalalignment='center'
    #              , verticalalignment='center'
    #              , bbox=dict(facecolor='white', alpha=0.7))
    #     plt.text(insample_boundary[0] // k_days, ax.get_ylim()[1], guide_date[1]
    #              , horizontalalignment='center'
    #              , verticalalignment='center'
    #              , bbox=dict(facecolor='white', alpha=0.7))
    #     plt.text(insample_boundary[1] // k_days, ax.get_ylim()[1], guide_date[2]
    #              , horizontalalignment='center'
    #              , verticalalignment='center'
    #              , bbox=dict(facecolor='white', alpha=0.7))
    #     plt.text(x[-1], ax.get_ylim()[1], guide_date[3]
    #              , horizontalalignment='center'
    #              , verticalalignment='center'
    #              , bbox=dict(facecolor='white', alpha=0.7))
    #
    # fig.savefig(os.path.join(outpath, 'test_y_{}{}.png'.format(ep, suffix)))
    # plt.close(fig)
    #
    # fig = plt.figure()
    # ax = plt.gca()
    # l_port_guide, = plt.plot(x, (1 + y_port - y_guide).cumprod() - 1.)
    # l_port_eq, = plt.plot(x,(1 + y_port - y_eq).cumprod() - 1.)
    # # l_port_base, = plt.plot(x, y_port - y_base)
    # # l_port_eq, = plt.plot(x, y_port - y_eq)
    # # l_label_base, = plt.plot(x, y_label - y_base)
    # # l_label_guide, = plt.plot(x, y_label - y_guide)
    #
    # # plt.legend(handles=(l_port_guide), labels=('port-guide'))
    # plt.legend(handles=(l_port_guide, l_port_eq), labels=('port-guide', 'port-eq'))
    #
    # # plt.legend(handles=(l_port_guide, l_port_base, l_port_eq, l_label_base, l_label_guide), labels=('port-guide','port-base', 'port-eq', 'label-base', 'label-guide'))
    # if insample_boundary is not None:
    #     plt.axvline(insample_boundary[0] / k_days)
    #     plt.axvline(insample_boundary[1] / k_days)
    #
    #     plt.text(x[0], ax.get_ylim()[1], guide_date[0]
    #              , horizontalalignment='center'
    #              , verticalalignment='center'
    #              , bbox=dict(facecolor='white', alpha=0.7))
    #     plt.text(insample_boundary[0] // k_days, ax.get_ylim()[1], guide_date[1]
    #              , horizontalalignment='center'
    #              , verticalalignment='center'
    #              , bbox=dict(facecolor='white', alpha=0.7))
    #     plt.text(insample_boundary[1] // k_days, ax.get_ylim()[1], guide_date[2]
    #              , horizontalalignment='center'
    #              , verticalalignment='center'
    #              , bbox=dict(facecolor='white', alpha=0.7))
    #     plt.text(x[-1], ax.get_ylim()[1], guide_date[3]
    #              , horizontalalignment='center'
    #              , verticalalignment='center'
    #              , bbox=dict(facecolor='white', alpha=0.7))
    #
    # fig.savefig(os.path.join(outpath, 'test_y_diff_{}{}.png'.format(ep, suffix)))
    # plt.close(fig)

    # fig = plt.figure()
    # plt.plot(x[:-1], active_share)
    # fig.savefig(os.path.join(outpath, 'test_activeshare_{}{}.png'.format(ep, suffix)))
    # plt.close(fig)


def plot(wgt_base, wgt_result, y_df_before, y_df_after, outpath='./out/'):

    n_asset = wgt_base.shape[1]
    viridis = cm.get_cmap('viridis', n_asset)

    # weight change
    x = np.arange(len(wgt_base))
    wgt_base_cum = wgt_base.cumsum(axis=1)
    fig = plt.figure()
    fig.suptitle('Weight Diff')
    ax1 = fig.add_subplot(211)
    for i in range(n_asset):
        if i == 0:
            ax1.fill_between(x, 0, wgt_base_cum[:, i], facecolor=viridis.colors[i], alpha=.7)
        else:
            ax1.fill_between(x, wgt_base_cum[:, i-1], wgt_base_cum[:, i], facecolor=viridis.colors[i], alpha=.7)

    wgt_result_cum = wgt_result.cumsum(axis=1)
    ax2 = fig.add_subplot(212)
    for i in range(n_asset):
        if i == 0:
            ax2.fill_between(x, 0, wgt_result_cum[:, i], facecolor=viridis.colors[i], alpha=.7)
        else:
            ax2.fill_between(x, wgt_result_cum[:, i-1], wgt_result_cum[:, i], facecolor=viridis.colors[i], alpha=.7)

    fig.savefig(os.path.join(outpath, 'test_wgt.png'))
    plt.close(fig)

    # features, labels = test_features, test_labels; wgt = wgt_result
    # active_share = np.sum(np.abs(wgt_result - wgt_base), axis=1)

    fig = plt.figure()
    y_df_before.index = pd.to_datetime(y_df_before.index)
    plt.plot((1 + y_df_before).cumprod(axis=0))
    plt.legend(labels=y_df_before.columns)

    plt.yscale('log')
    fig.savefig(os.path.join(outpath, 'test_y_before.png'))
    plt.close(fig)

    fig = plt.figure()
    y_df_after.index = pd.to_datetime(y_df_after.index)
    plt.plot((1 + y_df_after).cumprod(axis=0))
    plt.legend(labels=y_df_after.columns)

    plt.yscale('log')
    fig.savefig(os.path.join(outpath, 'test_y_after.png'))
    plt.close(fig)


def backtest(configs, sampler):
    c = configs

    wgt_base = np.zeros([sampler.max_len - c.base_i0 - 1, sampler.n_labels])
    wgt_result = np.zeros_like(wgt_base)

    # 20 days
    n_datapoint = c.retrain_days // c.k_days
    d, r = divmod(sampler.max_len - c.base_i0, c.retrain_days)

    y_next = np.zeros([d * n_datapoint + r // c.k_days + 1, sampler.n_labels])
    wgt_base_calc = np.zeros_like(y_next)
    wgt_result_calc = np.zeros_like(y_next)
    wgt_label_calc = np.zeros_like(y_next)

    wgt_date = list()

    # model & optimizer
    model = MyModel(sampler.n_features, sampler.n_labels, configs=c)
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=0.01)
    model.to(tu.device)

    ii = 0
    for t in range(c.base_i0, sampler.max_len, c.retrain_days):
        _, _, (dataset, _), _, _ = sampler.get_batch(t)
        features_prev, labels_prev, features, labels = tu.to_device(tu.device, to_torch(dataset))

        outpath_t = os.path.join(c.outpath, str(t))
        load_model(outpath_t, model, optimizer)

        with torch.set_grad_enabled(False):
            wgt_test, _, _, _, _ = model.forward_with_loss(features, None, mc_samples=c.mc_samples, loss_wgt=None, is_train=False)

        wgt_result[(t-c.base_i0):(t-c.base_i0+c.retrain_days), :] = tu.np_ify(wgt_test)[:c.retrain_days, :]
        wgt_base[(t-c.base_i0):(t-c.base_i0+c.retrain_days), :] = tu.np_ify(features['wgt'])[:c.retrain_days, :]

        y_next[ii:(ii + n_datapoint), :] = tu.np_ify(torch.exp(labels['logy_for_calc'])-1.)[::c.k_days, :][:n_datapoint, :]
        wgt_base_calc[ii:(ii + n_datapoint), :] = tu.np_ify(features['wgt'])[::c.k_days, :][:n_datapoint, :] # [:c.retrain_days:c.k_days, :]
        wgt_label_calc[ii:(ii + n_datapoint), :] = tu.np_ify(labels['wgt'])[::c.k_days, :][:n_datapoint, :] # [:c.retrain_days:c.k_days, :]

        wgt_result_calc[ii:(ii + n_datapoint), :] = tu.np_ify(wgt_test)[::c.k_days, :][:n_datapoint, :] # [:c.retrain_days:c.k_days, :]
        wgt_date += list(sampler.add_infos['date'][t::c.k_days][:n_datapoint])
        # schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=10, last_epoch=-1)
        ii += n_datapoint

    wgt_df = pd.DataFrame(data=wgt_result_calc, index=wgt_date, columns=sampler.add_infos['idx_list'])
    wgt_df.to_csv(os.path.join(c.outpath, 'wgt.csv'))
    y_date = wgt_date[:] + [sampler.add_infos['date'][-1]]

    active_share = np.sum(np.abs(wgt_result_calc - wgt_base_calc), axis=1)

    y_base, turnover_base = calc_y(wgt_base_calc, y_next, c.cost_rate)
    y_port, turnover_port = calc_y(wgt_result_calc, y_next, c.cost_rate)
    y_label, turnover_label = calc_y(wgt_label_calc, y_next, c.cost_rate)
    y_eq, turnover_eq = calc_y(np.repeat(np.ones([1, sampler.n_labels], dtype=np.float32) / sampler.n_labels, len(wgt_base_calc), axis=0), y_next, c.cost_rate)
    if c.base_weight is not None:
        y_guide, turnover_guide = calc_y(np.repeat(np.array([c.base_weight]), len(wgt_base_calc), axis=0), y_next, c.cost_rate)
    else:
        y_guide, turnover_guide = y_eq, turnover_eq

    y_df_before = pd.DataFrame(data={'base': y_base['before_cost'],
                              'port': y_port['before_cost'],
                              'label': y_label['before_cost'],
                              'eq': y_eq['before_cost'],
                              'guide': y_guide['before_cost']}, index=y_date)
    y_df_before.to_csv(os.path.join(c.outpath, 'y_before_cost.csv'))

    y_df_after = pd.DataFrame(data={'base': y_base['after_cost'],
                              'port': y_port['after_cost'],
                              'label': y_label['after_cost'],
                              'eq': y_eq['after_cost'],
                              'guide': y_guide['after_cost']}, index=y_date)
    y_df_after.to_csv(os.path.join(c.outpath, 'y_after_cost.csv'))

    plot(wgt_base, wgt_result, y_df_before, y_df_after, c.outpath)


def train(configs, model, optimizer, sampler, t=None):
    c = configs

    if t is None:
        iter_ = range(c.base_i0, sampler.max_len, c.retrain_days)
        t0 = c.base_i0
    else:
        iter_ = range(t, t+1)
        t0 = t

    for t in iter_:
        # ################ config in loop ################
        outpath_t = os.path.join(c.outpath, str(t))
        if os.path.isdir(outpath_t):
            r = re.compile("{}+".format(t))
            n = len(list(filter(r.match, os.listdir(c.outpath))))
            outpath_t = outpath_t + "_{}".format(n)

        os.makedirs(outpath_t, exist_ok=True)

        str_ = c.export()
        with open(os.path.join(outpath_t, 'c.txt'), 'w') as f:
            f.write(str_)

        adaptive_flag = c.init_loop()

        x_plot = np.arange(c.num_epochs)
        losses_dict = {'train': np.zeros([c.num_epochs]), 'eval': np.zeros([c.num_epochs]), 'test': np.zeros([c.num_epochs])}

        if c.model_init_everytime:
            # ################ model & optimizer in loop  # load 무시, 매 타임 초기화 ################
            model = MyModel(sampler.n_features, sampler.n_labels, configs=c)
            model.train()
            model.to(tu.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=0.01)
        else:
            model.train()
            if t > t0:  # 최초 한번만 가이드 따라가기
                c.adaptive_count = 0
                c.n_pretrain = 0

        # schedule = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=c.lr, max_lr=1e-2, gamma=1, last_epoch=-1)
        # schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1/10, last_epoch=-1)

        # ################ data processing in loop ################
        dataset_cpu, dataset = {}, {}
        f_prev_dict, l_prev_dict, f_dict, l_dict = {}, {}, {}, {}
        dataset_cpu['train'], dataset_cpu['eval'], dataset_cpu['test'], (dataset_cpu['test_insample'], insample_boundary), guide_date = sampler.get_batch(t)
        for key in dataset_cpu.keys():
            dataset[key] = deepcopy(dataset_cpu[key][0])
            dataset[key] = tu.to_device(tu.device, to_torch(dataset[key]))

            if key == 'eval':
                f_prev_dict[key], l_prev_dict[key], f_dict[key], l_dict[key] = dataset[c.es_type]
            else:
                f_prev_dict[key], l_prev_dict[key], f_dict[key], l_dict[key] = dataset[key]

        # eval_features_prev, eval_features, eval_labels = dataset[c.es_type]
        # test_features_prev, test_features, test_labels = dataset['test']
        # testin_features_prev, testin_features, testin_labels = dataset['test_insample']

        loss_threshold = c.loss_threshold
        for ep in range(c.num_epochs):
            # if ep > 0:
            #     print(losses_train_np, losses_eval, losses_test)
            if ep % c.eval_freq == 0:
                with torch.set_grad_enabled(False):
                    _, losses_eval, _, _, losses_eval_dict = model.forward_with_loss(f_dict['eval'], l_dict['eval']
                                                                                     , mc_samples=c.mc_samples
                                                                                     , loss_wgt=c.loss_wgt
                                                                                     , features_prev=f_prev_dict['eval']
                                                                                     , labels_prev=l_prev_dict['eval'])

                    losses_eval = tu.np_ify(losses_eval)

                    losses_dict['eval'][ep] = float(losses_eval)

                if ep >= c.n_pretrain * c.eval_freq:
                    if losses_eval > c.min_eval_loss:
                        c.es_count += 1
                    else:
                        model.save_to_optim()
                        c.min_eval_loss = losses_eval
                        c.es_count = 0

                str_ = ""
                for key in losses_eval_dict.keys():
                    str_ += "{}: {:2.2f} / ".format(key, float(tu.np_ify(losses_eval_dict[key]) * c.loss_wgt[key]))

                _, losses_test, _, _, loss_test_dict = model.forward_with_loss(f_dict['test'], l_dict['test']
                                                                               , mc_samples=c.mc_samples
                                                                               , loss_wgt=c.loss_wgt
                                                                               , features_prev=f_prev_dict['test']
                                                                               , labels_prev=l_prev_dict['test']
                                                                               , is_train=False)

                losses_dict['test'][ep] = float(losses_test)
                save_model(outpath_t, ep, model, optimizer)
                suffix = "_e[{:2.2f}]test[{:2.2f}]".format(losses_eval, losses_test)
                if ep % (c.plot_freq * c.eval_freq) == 0:
                    plot_each(ep, sampler, model, f_dict['test_insample'], l_dict['test_insample']
                              , insample_boundary=insample_boundary
                              , guide_date=guide_date
                              , mc_samples=c.mc_samples
                              , k_days=c.k_days
                              , cost_rate=c.cost_rate
                              , suffix=suffix
                              , outpath=outpath_t
                              , guide_weight=c.base_weight)

                if ep > 0 and ep % (c.plot_freq * c.eval_freq) == 0:
                    x_len = (ep // 10000 + 1) * 10000
                    sampling_freq = (ep // 10000 + 1) * 1000
                    fig = plt.figure()
                    l_train, = plt.plot(x_plot[1:x_len], losses_dict['train'][1:x_len])
                    l_eval, = plt.plot(x_plot[1:x_len], losses_dict['eval'][1:x_len])
                    loss_test = np.zeros_like(losses_dict['test'][:x_len])
                    loss_test[::sampling_freq] = losses_dict['test'][:x_len][::sampling_freq]
                    l_test, = plt.plot(x_plot[1:x_len], loss_test[1:])

                    plt.legend(handles=(l_train, l_eval, l_test), labels=('train', 'eval', 'test'))

                    fig.savefig(os.path.join(outpath_t, 'learning_curve_{}.png'.format(t)))
                    plt.close(fig)

                # if adaptive_flag and c.es_count >= c.adaptive_count:
                if adaptive_flag and ((loss_threshold is not None and losses_eval < loss_threshold) or c.es_count >= c.adaptive_count):
                    print('[changed] {} {} e {:3.2f} / {}'.format(t, ep, losses_eval, str_))
                    adaptive_flag = False
                    loss_threshold = c.adaptive_loss_threshold
                    optimizer.param_groups[0]['lr'] = c.lr * c.adaptive_lrx
                    # optimizer.param_groups[0]['lr'] = c.lr * 10

                    c.es_max = c.es_max_count
                    c.es_count = 0;
                    c.min_eval_loss = 99999

                    for key in losses_eval_dict.keys():
                        if c.adaptive_loss_wgt[key] >= 0:
                            c.loss_wgt[key] = c.adaptive_loss_wgt[key]

                        if c.loss_wgt[key] == 0:
                            continue

                        val = np.abs(tu.np_ify(losses_eval_dict[key]))
                        if c.loss_wgt[key] < 0 and val > 10:
                            # if val > 10:
                            c.loss_wgt[key] = 10. / float(val * abs(c.loss_wgt[key]))

                    continue

                if (loss_threshold is not None and losses_eval < loss_threshold) or (c.es_max > 0 and c.es_count >= c.es_max):
                    print('[stopped] {} {} e {:3.2f} / {}'.format(t, ep, losses_eval, str_))
                    break

            train_dataloader = data_loader(dataset_cpu['train'], batch_size=c.batch_size)
            for i_loop, (train_f_prev, train_l_prev, train_f, train_l) in enumerate(train_dataloader):
                """
                train_f_prev, train_l_prev, train_f, train_l = next(iter(train_dataloader))
                """
                train_f_prev, train_l_prev, train_f, train_l = tu.to_device(tu.device, [train_f_prev, train_l_prev, train_f, train_l])

                if c.adv_train is True:
                    train_f = model.adversarial_noise(train_f, train_l, features_prev=train_f_prev, labels_prev=train_l_prev)

                _, losses_train, _, _, _ = model.forward_with_loss(train_f, train_l
                                                                   , mc_samples=c.mc_samples
                                                                   , loss_wgt=c.loss_wgt
                                                                   , features_prev=train_f_prev
                                                                   , labels_prev=train_l_prev)

                # schedule.step()
                losses_train_np = tu.np_ify(losses_train)
                losses_dict['train'][ep] += float(losses_train_np)

                optimizer.zero_grad()
                losses_train.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip)
                optimizer.step()

            if ep % c.eval_freq == 0:
                print('{} {} t {:3.2f} / e {:3.2f} / {}'.format(t, ep, losses_dict['train'][ep], losses_eval, str_))

        model.load_from_optim()
        plot_each(c.num_epochs + 20000, sampler, model, f_dict['test_insample'], l_dict['test_insample']
                  , insample_boundary=insample_boundary, guide_date=guide_date
                  , mc_samples=c.mc_samples, k_days=c.k_days
                  , cost_rate=c.cost_rate
                  , suffix=suffix + "_{}".format(t), outpath=outpath_t
                  , guide_weight=c.base_weight)

        plot_each(c.num_epochs + 20000, sampler, model, f_dict['test'], l_dict['test']
                  , mc_samples=c.mc_samples, k_days=c.k_days
                  , cost_rate=c.cost_rate
                  , suffix=suffix + "_{}_test".format(t), outpath=outpath_t
                  , guide_weight=c.base_weight)


testmode = False
@profile
def main(testmode=False):

    base_weight = dict(h=[0.69, 0.2, 0.1, 0.01],
                       m=[0.4, 0.15, 0.075, 0.375],
                       l=[0.25, 0.1, 0.05, 0.6],
                       eq=[0.25, 0.25, 0.25, 0.25])

    for key in ['eq', 'l','m','h',]:
    # configs & variables
        name = 'apptest_new_12_{}'.format(key)
        # name = 'app_adv_1'
        c = Configs(name)
        c.num_epochs = 3000
        c.base_weight = base_weight[key]

        c.adaptive_loss_wgt = {'y_pf': 1, 'mdd_pf': 1000., 'logy': 1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.001,
                              'cost': 1., 'entropy': 0.0001}

        c.es_max_count = -1
        str_ = c.export()
        with open(os.path.join(c.outpath, 'c.txt'), 'w') as f:
            f.write(str_)

        # data processing
        features_dict, labels_dict, add_info = get_data(configs=c)
        sampler = Sampler(features_dict, labels_dict, add_info, configs=c)

        # model & optimizer
        model = MyModel(sampler.n_features, sampler.n_labels, configs=c)
        optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=0.1)
        load_model(c.outpath, model, optimizer)
        model.train()
        model.to(tu.device)

        train(c, model, optimizer, sampler, 3000)
        # train(c, model, optimizer, sampler, t=1700)

        backtest(c, sampler)

    # # ####### plot test
    # for ii, t in enumerate(range(base_i, sampler.max_len, rebal_freq)):
    #     train_dataset, eval_dataset, test_dataset, (test_dataset_insample, insample_boundary) = sampler.get_batch(t)
    #     test_features_insample, test_labels_insample = to_torch(test_dataset_insample)
    #
    #     test_features_insample, test_labels_insample = tu.to_device(tu.device, [test_features_insample, test_labels_insample])
    #
    #     plot_each(0, model, test_features_insample, test_labels_insample, insample_boundary=insample_boundary,
    #               n_samples=n_samples, rebal_freq=rebal_freq, suffix=t, outpath=outpath)


def train_anp(dataset, model, optimizer, is_train):

    if is_train:
        iter_ = 100
        model.train()
    else:
        iter_ = 1
        model.eval()

    losses = 0
    for it in range(iter_):
        batch_dataset = random.sample(dataset, 64)  # batch_size
        c_x = np.stack([batch_dataset[batch_i][0] for batch_i in range(64)])
        c_y = np.stack([batch_dataset[batch_i][1] for batch_i in range(64)])
        t_x = np.stack([batch_dataset[batch_i][2] for batch_i in range(64)])
        t_y = np.stack([batch_dataset[batch_i][3] for batch_i in range(64)])

        c_x = torch.from_numpy(c_x).float().to(tu.device)
        c_y = torch.from_numpy(c_y).float().to(tu.device)
        t_x = torch.from_numpy(t_x).float().to(tu.device)
        t_y = torch.from_numpy(t_y).float().to(tu.device)

        query = (c_x, c_y), t_x
        target_y = t_y

        with torch.set_grad_enabled(is_train):
            mu, sigma, log_p, global_kl, local_kl, loss = model(query, target_y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        losses += tu.np_ify(loss)

    losses = losses / iter_
    return losses


# @profile
def plot_functions(path, ep, base_i, sampler, model):
    """Plots the predicted mean and variance and the context points.

    Args:
        target_x: An array of shape [B,num_targets,1] that contains the
            x values of the target points.
        target_y: An array of shape [B,num_targets,1] that contains the
            y values of the target points.
        context_x: An array of shape [B,num_contexts,1] that contains
            the x values of the context points.
        context_y: An array of shape [B,num_contexts,1] that contains
            the y values of the context points.
        pred_y: An array of shape [B,num_targets,1] that contains the
            predicted means of the y values at the target points in target_x.
        std: An array of shape [B,num_targets,1] that contains the
            predicted std dev of the y values at the target points in target_x.
    """

    dataset = sampler.get_batch_set(base_i, is_train=False)
    batch_dataset = dataset[::-20]
    batch_dataset.reverse()
    c_x = np.stack([batch_dataset[batch_i][0] for batch_i in range(len(batch_dataset))])
    c_y = np.stack([batch_dataset[batch_i][1] for batch_i in range(len(batch_dataset))])
    t_x = np.stack([batch_dataset[batch_i][2] for batch_i in range(len(batch_dataset))])
    t_y = np.stack([batch_dataset[batch_i][3] for batch_i in range(len(batch_dataset))])

    c_x = torch.from_numpy(c_x).float().to(tu.device)
    c_y = torch.from_numpy(c_y).float().to(tu.device)
    t_x = torch.from_numpy(t_x).float().to(tu.device)
    t_y = torch.from_numpy(t_y).float().to(tu.device)

    query = (c_x, c_y), t_x
    target_y = t_y

    with torch.set_grad_enabled(False):
        mu, sigma, log_p, global_kl, local_kl, loss = model(query, None)

    t_sample = t_x[0, :, -1]



    # t = t_x[0, :, -1]
    t = np.arange(len(mu) + 1)
    pred_logy = tu.np_ify(mu[:, -1, :])
    pred_sigma = tu.np_ify(sigma[:, -1, :])
    label_logy = tu.np_ify(target_y[:, -1, :])

    pred_logy_cum = np.concatenate([np.zeros([1, pred_logy.shape[-1]]), pred_logy.cumsum(axis=0)], axis=0)
    label_logy_cum = np.concatenate([np.zeros([1, label_logy.shape[-1]]), label_logy.cumsum(axis=0)], axis=0)
    pred_sigma_cum = np.concatenate([np.zeros([1, pred_sigma.shape[-1]]), pred_sigma], axis=0)

    fig = plt.figure()
    ax = []
    for plot_i in range(4):
        ax.append(fig.add_subplot(2, 2, plot_i+1))
        # Plot everything
        ax[plot_i].plot(t[:-1], pred_logy[:, plot_i], 'b', linewidth=2)
        ax[plot_i].plot(t[:-1], label_logy[:, plot_i], 'k:', linewidth=2)

        plt.fill_between(
            t[:-1],
            pred_logy[:, plot_i] - pred_sigma[:, plot_i],
            pred_logy[:, plot_i] + pred_sigma[:, plot_i],
            alpha=0.2,
            facecolor='#65c9f7',
            interpolate=True)
        plt.grid('off')

    file_path = os.path.join(path, 'test_{}.png'.format(ep))
    fig.savefig(file_path)
    plt.close(fig)


    fig = plt.figure()
    ax = []
    for plot_i in range(4):
        ax.append(fig.add_subplot(2, 2, plot_i+1))
        # Plot everything
        ax[plot_i].plot(t, pred_logy_cum[:, plot_i], 'b', linewidth=2)
        ax[plot_i].plot(t, label_logy_cum[:, plot_i], 'k:', linewidth=2)

        plt.fill_between(
            t,
            pred_logy_cum[:, plot_i] - pred_sigma_cum[:, plot_i],
            pred_logy_cum[:, plot_i] + pred_sigma_cum[:, plot_i],
            alpha=0.2,
            facecolor='#65c9f7',
            interpolate=True)
        plt.grid('off')

    file_path = os.path.join(path, 'test_cum_{}.png'.format(ep))
    fig.savefig(file_path)
    plt.close(fig)


def main_anp():
    from model_anp import LatentModel
    from data_anp import get_data as get_data_anp, Sampler as Sampler_anp
    # name = 'apptest_adv_28'
    name = 'anp_2'
    c = Configs(name)

    str_ = c.export()
    with open(os.path.join(c.outpath, 'c.txt'), 'w') as f:
        f.write(str_)

    # data processing
    features_dict, labels_dict, add_infos = get_data_anp(configs=c)
    sampler = Sampler_anp(features_dict, labels_dict, add_infos, configs=c)

    # model & optimizer
    model = LatentModel(32, sampler.n_features, sampler.n_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=0.01)
    load_model(c.outpath, model, optimizer)
    model.train()
    model.to(tu.device)

    min_eval_loss = 99999
    earlystop_count = 0
    ep = 0
    base_i = 2500
    dataset = sampler.get_batch_set(base_i)
    while ep < c.num_epochs:
        eval_loss = train_anp(dataset, model, optimizer, is_train=False)
        if min_eval_loss > eval_loss:
            model.save_to_optim()
            min_eval_loss = eval_loss
            earlystop_count = 0
        else:
            earlystop_count += 1

        print("[base_i: {}, ep: {}] eval_loss: {} / count: {}".format(base_i, ep, eval_loss, earlystop_count))
        if earlystop_count >= 5:
            model.load_from_optim()
            plot_functions(c.outpath, ep, base_i + 500, sampler, model)

            min_eval_loss = 99999
            earlystop_count = 0
            break

        if ep % 5 == 0:
            plot_functions(c.outpath, ep, base_i, sampler, model)

        train_loss = train_anp(dataset, model, optimizer, is_train=True)
        ep += 1


def test():
    adaptive_lrx_l = [2, 5, 10]  # learning rate * 배수
    use_accum_data_l = [True, False]  # [sampler] 데이터 누적할지 말지

    random_guide_weight_l = [0., 0.2, 0.5, 0.8, 1.]
    adaptive_loss_wgt_l = [
        {'y_pf': 0.2, 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.01, 'cost': 1., 'entropy': 0.0001}]
    # adaptive_loss_wgt_l = [{'y_pf': 0.2, 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.05, 'cost': 1., 'entropy': 0.0001}
    #                      , {'y_pf': 0.2, 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.05, 'cost': 1., 'entropy': 0.001}
    #                      , {'y_pf': 0.2, 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.05, 'cost': 1., 'entropy': 0.01}
    #                      , {'y_pf': 0.2, 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.05, 'cost': 1., 'entropy': 0.1}
    #
    #                      , {'y_pf': 0.2, 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.01, 'cost': 1., 'entropy': 0.001}
    #                      , {'y_pf': 0.2, 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.05, 'cost': 1., 'entropy': 0.001}
    #                      , {'y_pf': 0.2, 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.1, 'cost': 1., 'entropy': 0.001}
    #                      , {'y_pf': 0.2, 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.5, 'cost': 1., 'entropy': 0.001}
    #                      , {'y_pf': 0.2, 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 1, 'cost': 1., 'entropy': 0.001}
    #
    #                      , {'y_pf': 0.2, 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.05, 'cost': 1., 'entropy': 0.001}
    #                      , {'y_pf': 0.2, 'mdd_pf': 100., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.05, 'cost': 1., 'entropy': 0.001}
    #                      , {'y_pf': 0.2, 'mdd_pf': 10., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.05, 'cost': 1., 'entropy': 0.001}
    #                      , {'y_pf': 0.2, 'mdd_pf': 1., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.05, 'cost': 1., 'entropy': 0.001}
    #
    #                      , {'y_pf': 0.2, 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.05, 'cost': 1., 'entropy': 0.001}
    #                      , {'y_pf': 0.2, 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.05, 'cost': 0., 'entropy': 0.001}
    #
    #                      , {'y_pf': 0., 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.05, 'cost': 0., 'entropy': 0.001}
    #                      , {'y_pf': 0.2, 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.05, 'cost': 0., 'entropy': 0.001}
    #                      , {'y_pf': 0.5, 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.05, 'cost': 0., 'entropy': 0.001}
    #                      , {'y_pf': 1., 'mdd_pf': 1000., 'logy': -1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.05, 'cost': 0., 'entropy': 0.001}
    #                      ]
    for adaptive_lrx in adaptive_lrx_l:
        for use_accum_data in use_accum_data_l:
            for t in [3600, 3000, 1500, ]:
                for random_guide_weight in random_guide_weight_l:
                    for adaptive_loss_wgt in adaptive_loss_wgt_l:
                        name = 'app_adv_5'
                        c = Configs(name)

                        str_ = c.export()
                        with open(os.path.join(c.outpath, 'c.txt'), 'w') as f:
                            f.write(str_)

                        c.adaptive_lrx = adaptive_lrx
                        c.use_accum_data = use_accum_data
                        c.random_guide_weight = random_guide_weight
                        c.adaptive_loss_wgt = adaptive_loss_wgt

                        # data processing
                        features_dict, labels_dict, add_info = get_data(configs=c)
                        sampler = Sampler(features_dict, labels_dict, add_info, configs=c)

                        # model & optimizer
                        model = MyModel(sampler.n_features, sampler.n_labels, configs=c)
                        optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=0.01)
                        load_model(c.outpath, model, optimizer)
                        model.train()
                        model.to(tu.device)

                        train(c, model, optimizer, sampler, t)

# if __name__ == '__main__':
#     main()