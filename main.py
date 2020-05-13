
import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt, cm


from model import MyModel, load_model, save_model
from data import get_data, Sampler, to_torch
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
        self.num_epochs = 10000
        self.base_i0 = 1000
        self.n_samples = 200
        self.k_days = 20
        self.label_days = 20
        self.strategy_days = 250
        self.adaptive_count = 10
        self.es_max_count = 50
        self.retrain_days = 60
        self.test_days = 500  # test days
        self.init_train_len = 500
        self.normalizing_window = 500  # norm windows for macro data
        self.adaptive_flag = True
        self.n_pretrain = 20

        self.datatype = 'app'
        self.init_weight()

        self.cost_rate = 0.003

        self.plot_freq = 1000

        self.set_path()

    def init_weight(self):
        if self.datatype == 'app':
            self.cash_idx = 3
            self.base_weight = [0.699, 0.2, 0.1, 0.001]
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


def plot_each(ep, model, features, labels, insample_boundary=None, guide_date=None, n_samples=100, k_days=20, suffix='', outpath='./out/'):
    # ep=0; n_samples = 100; k_days = 20; model, features, labels, insample_boundary = main(testmode=True)

    with torch.set_grad_enabled(False):
        wgt_test, losses_test, pred_mu_test, pred_sigma_test, _ = model.forward_with_loss(features, None, n_samples=n_samples)

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
    ax1 = fig.add_subplot(311)
    ax1.set_title('base')
    for i in range(n_asset):
        if i == 0:
            ax1.fill_between(x, 0, wgt_base_cum[:, i], facecolor=viridis.colors[i], alpha=.7)
        else:
            ax1.fill_between(x, wgt_base_cum[:, i-1], wgt_base_cum[:, i], facecolor=viridis.colors[i], alpha=.7)

    wgt_result_cum = wgt_result.cumsum(axis=1)
    ax2 = fig.add_subplot(312)
    ax2.set_title('result')
    for i in range(n_asset):
        if i == 0:
            ax2.fill_between(x, 0, wgt_result_cum[:, i], facecolor=viridis.colors[i], alpha=.7)
        else:
            ax2.fill_between(x, wgt_result_cum[:, i-1], wgt_result_cum[:, i], facecolor=viridis.colors[i], alpha=.7)

    wgt_label_cum = wgt_label.cumsum(axis=1)
    ax3 = fig.add_subplot(313)
    ax3.set_title('label')
    for i in range(n_asset):
        if i == 0:
            ax3.fill_between(x, 0, wgt_label_cum[:, i], facecolor=viridis.colors[i], alpha=.7)
        else:
            ax3.fill_between(x, wgt_label_cum[:, i-1], wgt_label_cum[:, i], facecolor=viridis.colors[i], alpha=.7)

    if insample_boundary is not None:
        ax1.axvline(insample_boundary[0])
        ax1.axvline(insample_boundary[1])

        ax2.axvline(insample_boundary[0])
        ax2.axvline(insample_boundary[1])

        ax3.axvline(insample_boundary[0])
        ax3.axvline(insample_boundary[1])

        ax1.text(x[0], ax1.get_ylim()[1], guide_date[0]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))
        ax1.text(insample_boundary[0], ax1.get_ylim()[1], guide_date[1]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))
        ax1.text(insample_boundary[1], ax1.get_ylim()[1], guide_date[2]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))
        ax1.text(x[-1], ax1.get_ylim()[1], guide_date[3]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))

    fig.savefig(os.path.join(outpath, 'test_wgt_{}{}.png'.format(ep, suffix)))
    plt.close(fig)

    y_next = tu.np_ify(labels['logy_for_calc'])[::k_days, :]
    wgt_base_calc = wgt_base[::k_days, :]
    wgt_result_calc = wgt_result[::k_days, :]
    wgt_label_calc = wgt_label[::k_days, :]
    # date_base.append(sampler.add_infos['date'][t])

    # features, labels = test_features, test_labels; wgt = wgt_result
    # active_share = np.sum(np.abs(wgt_result - wgt_base), axis=1)
    active_share = np.sum(np.abs(wgt_result_calc - wgt_base_calc), axis=1)

    y_base = np.sum((1+y_next) * wgt_base_calc, axis=1).cumprod()
    y_port = np.sum((1+y_next) * wgt_result_calc, axis=1).cumprod()
    y_label = np.sum((1+y_next) * wgt_label_calc, axis=1).cumprod()
    y_eq = np.mean(1+y_next, axis=1).cumprod()
    y_guide = np.sum((1+y_next) * np.array([0.7, 0.2, 0.1, 0.0]), axis=1).cumprod()

    y_base = y_base / y_base[0]
    y_port = y_port / y_port[0]
    y_label = y_label / y_label[0]
    y_eq = y_eq / y_eq[0]
    y_guide = y_guide / y_guide[0]

    x = np.arange(len(y_base))

    fig = plt.figure()
    ax = plt.gca()
    l_base, = plt.plot(x, y_base)
    l_port, = plt.plot(x, y_port)
    l_label, = plt.plot(x, y_label)
    l_eq, = plt.plot(x, y_eq)
    l_guide, = plt.plot(x, y_guide)
    plt.legend(handles=(l_base, l_port, l_label, l_eq, l_guide), labels=('base', 'port', 'label', 'eq', 'guide'))
    if insample_boundary is not None:
        plt.axvline(insample_boundary[0] / k_days)
        plt.axvline(insample_boundary[1] / k_days)

        plt.text(x[0], ax.get_ylim()[1], guide_date[0]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))
        plt.text(insample_boundary[0] // k_days, ax.get_ylim()[1], guide_date[1]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))
        plt.text(insample_boundary[1] // k_days, ax.get_ylim()[1], guide_date[2]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))
        plt.text(x[-1], ax.get_ylim()[1], guide_date[3]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))

    fig.savefig(os.path.join(outpath, 'test_y_{}{}.png'.format(ep, suffix)))
    plt.close(fig)

    fig = plt.figure()
    ax = plt.gca()
    l_port_guide, = plt.plot(x, y_port - y_guide)
    l_port_base, = plt.plot(x, y_port - y_base)
    l_port_eq, = plt.plot(x, y_port - y_eq)
    l_label_base, = plt.plot(x, y_label - y_base)
    l_label_guide, = plt.plot(x, y_label - y_guide)

    plt.legend(handles=(l_port_guide, l_port_base, l_port_eq, l_label_base, l_label_guide), labels=('port-guide','port-base', 'port-eq', 'label-base', 'label-guide'))
    if insample_boundary is not None:
        plt.axvline(insample_boundary[0] / k_days)
        plt.axvline(insample_boundary[1] / k_days)

        plt.text(x[0], ax.get_ylim()[1], guide_date[0]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))
        plt.text(insample_boundary[0] // k_days, ax.get_ylim()[1], guide_date[1]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))
        plt.text(insample_boundary[1] // k_days, ax.get_ylim()[1], guide_date[2]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))
        plt.text(x[-1], ax.get_ylim()[1], guide_date[3]
                 , horizontalalignment='center'
                 , verticalalignment='center'
                 , bbox=dict(facecolor='white', alpha=0.7))

    fig.savefig(os.path.join(outpath, 'test_y_diff_{}{}.png'.format(ep, suffix)))
    plt.close(fig)

    fig = plt.figure()
    plt.plot(x, active_share)
    fig.savefig(os.path.join(outpath, 'test_activeshare_{}{}.png'.format(ep, suffix)))
    plt.close(fig)


def plot(wgt_base, wgt_result, y_df_after, outpath='./out/'):

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

    y_df_after.index = pd.to_datetime(y_df_after.index)
    plt.plot((1 + y_df_after).cumprod(axis=0))
    plt.legend(labels=y_df_after.columns)

    plt.yscale('log')
    fig.savefig(os.path.join(outpath, 'test_y.png'))
    plt.close(fig)


def backtest(configs, sampler):
    c = configs

    wgt_base = np.zeros([sampler.max_len - c.base_i0, sampler.n_labels])
    wgt_result = np.zeros_like(wgt_base)

    # 20 days
    y_next = np.zeros([(sampler.max_len - c.base_i0) // c.k_days + 1, sampler.n_labels])
    wgt_base_calc = np.zeros_like(y_next)
    wgt_result_calc = np.zeros_like(y_next)
    wgt_label_calc = np.zeros_like(y_next)

    wgt_date = list()

    # model & optimizer
    model = MyModel(sampler.n_features, sampler.n_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=0.01)
    model.to(tu.device)

    ii = 0
    for t in range(c.base_i0, sampler.max_len, c.retrain_days):

        _, _, dataset, _, _ = sampler.get_batch(t)
        features, labels = tu.to_device(tu.device, to_torch(dataset))

        outpath_t = os.path.join(c.outpath, str(t))
        load_model(outpath_t, model, optimizer)

        with torch.set_grad_enabled(False):
            wgt_test, _, _, _, _ = model.forward_with_loss(features, None, n_samples=c.n_samples, loss_wgt=None)

        wgt_result[(t-c.base_i0):(t-c.base_i0+c.retrain_days), :] = tu.np_ify(wgt_test)[:c.retrain_days, :]
        wgt_base[(t-c.base_i0):(t-c.base_i0+c.retrain_days), :] = tu.np_ify(features['wgt'])[:c.retrain_days, :]

        n_datapoint = c.retrain_days // c.k_days
        y_next[ii:(ii + n_datapoint), :] = tu.np_ify(labels['logy_for_calc'])[:c.retrain_days:c.k_days, :]
        wgt_base_calc[ii:(ii + n_datapoint), :] = tu.np_ify(features['wgt'])[:c.retrain_days:c.k_days, :]
        wgt_label_calc[ii:(ii + n_datapoint), :] = tu.np_ify(labels['wgt'])[:c.retrain_days:c.k_days, :]

        wgt_result_calc[ii:(ii + n_datapoint), :] = tu.np_ify(wgt_test)[:c.retrain_days:c.k_days, :]
        wgt_date += list(sampler.add_infos['date'][t:(t+c.retrain_days):c.k_days])
        # schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=10, last_epoch=-1)
        ii += n_datapoint

    wgt_df = pd.DataFrame(data=wgt_result_calc, index=wgt_date, columns=sampler.add_infos['idx_list'])
    wgt_df.to_csv(os.path.join(c.outpath, 'wgt.csv'))
    y_date = wgt_date[:] + [sampler.add_infos['date'][-1]]

    active_share = np.sum(np.abs(wgt_result_calc - wgt_base_calc), axis=1)

    y_base, turnover_base = calc_y(wgt_base_calc, y_next, c.cost_rate)
    y_port, turnover_port = calc_y(wgt_result_calc, y_next, c.cost_rate)
    y_label, turnover_label = calc_y(wgt_label_calc, y_next, c.cost_rate)
    y_eq, turnover_eq = calc_y(np.repeat(np.array([[0.25, 0.25, 0.25, 0.25]]), len(wgt_base_calc), axis=0), y_next, c.cost_rate)
    y_guide, turnover_guide = calc_y(np.repeat(np.array([[0.7, 0.2, 0.1, 0.0]]), len(wgt_base_calc), axis=0), y_next, c.cost_rate)

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

    plot(wgt_base, wgt_result, y_df_after, c.outpath)


def train(configs, model, optimizer, sampler, t=None, adv_train=False):
    c = configs

    if t is None:
        iter_ = range(c.base_i0, sampler.max_len, c.retrain_days)
    else:
        iter_ = range(t, t+1)
    for t in iter_:
        # ################ config in loop ################
        outpath_t = os.path.join(c.outpath, str(t))
        os.makedirs(outpath_t, exist_ok=True)
        adaptive_flag = c.init_loop()

        dataset = {'train': None, 'eval': None, 'test': None, 'test_insample': None}

        x_plot = np.arange(c.num_epochs)
        losses_dict = {'train': np.zeros([c.num_epochs]), 'eval': np.zeros([c.num_epochs]), 'test': np.zeros([c.num_epochs])}

        # ################ data processing in loop ################
        dataset['train'], dataset['eval'], dataset['test'], (dataset['test_insample'], insample_boundary), guide_date = sampler.get_batch(t)

        for key in dataset.keys():
            dataset[key] = tu.to_device(tu.device, to_torch(dataset[key]))

        # ################ model & optimizer in loop  # load 무시, 매 타임 초기화 ################
        model = MyModel(sampler.n_features, sampler.n_labels)
        model.train()
        model.to(tu.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=0.01)
        # schedule = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=c.lr, max_lr=1e-2, gamma=1, last_epoch=-1)
        # schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1/10, last_epoch=-1)

        for ep in range(c.num_epochs):
            train_features, train_labels = dataset['train']
            if adv_train is True:
                train_features = model.adversarial_noise(train_features, train_labels)

            _, losses_train, _, _, _ = model.forward_with_loss(train_features, train_labels
                                                               , n_samples=c.n_samples
                                                               , loss_wgt=c.loss_wgt)

            optimizer.zero_grad()
            losses_train.backward()
            optimizer.step()
            # schedule.step()
            losses_train = tu.np_ify(losses_train)
            losses_dict['train'][ep] = float(losses_train)

            with torch.set_grad_enabled(False):
                eval_features, eval_labels = dataset['eval']
                _, losses_eval, _, _, losses_eval_dict = model.forward_with_loss(eval_features, eval_labels
                                                                                 , n_samples=c.n_samples
                                                                                 , loss_wgt=c.loss_wgt)

                losses_eval = tu.np_ify(losses_eval)

                losses_dict['eval'][ep] = float(losses_eval)

            if ep % 100 == 0:
                if ep >= c.n_pretrain * 100:
                    if losses_eval > c.min_eval_loss:
                        c.es_count += 1
                    else:
                        model.save_to_optim()
                        c.min_eval_loss = losses_eval
                        c.es_count = 0

                str_ = ""
                for key in losses_eval_dict.keys():
                    str_ += "{}: {:2.2f} / ".format(key, float(tu.np_ify(losses_eval_dict[key]) * c.loss_wgt[key]))

                print('{} {} t {:3.2f} / e {:3.2f} / {}'.format(t, ep, losses_train, losses_eval, str_))

                _, losses_test, _, _, loss_test_dict = model.forward_with_loss(dataset['test'][0], dataset['test'][1]
                                                                               , n_samples=c.n_samples
                                                                               , loss_wgt=c.loss_wgt)

                losses_dict['test'][ep] = float(losses_test)
                save_model(outpath_t, ep, model, optimizer)
                suffix = "_t[{:2.2f}]e[{:2.2f}]test[{:2.2f}]".format(losses_train, losses_eval, losses_test)
                if ep % c.plot_freq == 0:
                    plot_each(ep, model, dataset['test_insample'][0], dataset['test_insample'][1]
                              , insample_boundary=insample_boundary
                              , guide_date=guide_date
                              , n_samples=c.n_samples
                              , k_days=c.k_days
                              , suffix=suffix
                              , outpath=outpath_t)

                if ep > 0:
                    x_len = (ep // 10000 + 1) * 10000
                    sampling_freq = (ep // 10000 + 1) * 100
                    fig = plt.figure()
                    l_train, = plt.plot(x_plot[:x_len], losses_dict['train'][:x_len])
                    l_eval, = plt.plot(x_plot[:x_len], losses_dict['eval'][:x_len])
                    loss_test = np.zeros_like(losses_dict['test'][:x_len])
                    loss_test[::sampling_freq] = losses_dict['test'][:x_len][::sampling_freq]
                    l_test, = plt.plot(x_plot[:x_len], loss_test)

                    plt.legend(handles=(l_train, l_eval, l_test), labels=('train', 'eval', 'test'))

                    fig.savefig(os.path.join(outpath_t, 'learning_curve_{}.png'.format(t)))
                    plt.close(fig)

                if c.es_count > c.adaptive_count and adaptive_flag:
                    adaptive_flag = False
                    optimizer.param_groups[0]['lr'] = c.lr * 10

                    c.es_max = c.es_max_count
                    c.es_count = 0; c.min_eval_loss = 99999

                    for key in losses_eval_dict.keys():
                        if key in ['entropy', 'cost', 'wgt_guide']:
                            c.loss_wgt[key] = 0.1
                        elif key in ['mdd_pf']:
                            c.loss_wgt[key] = 1000
                        elif key in ['wgt_guide', 'wgt2', 'wgt']:
                            c.loss_wgt[key] = 0

                        if c.loss_wgt[key] == 0:
                            continue
                        val = np.abs(tu.np_ify(losses_eval_dict[key]))
                        if val > 10:
                            # c.loss_wgt[key] = 0
                            c.loss_wgt[key] = 1. / float(val * c.loss_wgt[key])

                if c.es_max > 0 and c.es_count >= c.es_max:
                    model.load_from_optim()
                    plot_each(20000, model, dataset['test_insample'][0], dataset['test_insample'][1]
                              , insample_boundary=insample_boundary, guide_date=guide_date
                              , n_samples=c.n_samples, k_days=c.k_days
                              , suffix=suffix + "_{}".format(t), outpath=outpath_t)

                    break


testmode = False
@profile
def main(testmode=False):

    # configs & variables
    name = 'apptest_adv_7'
    c = Configs(name)

    str_ = c.export()
    with open(os.path.join(c.outpath, 'c.txt'), 'w') as f:
        f.write(str_)

    # data processing
    features_dict, labels_dict, add_info = get_data(configs=c)
    sampler = Sampler(features_dict, labels_dict, add_info, init_train_len=c.init_train_len, label_days=c.label_days, test_days=c.test_days)

    # model & optimizer
    model = MyModel(sampler.n_features, sampler.n_labels, cost_rate=c.cost_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=0.01)
    load_model(c.outpath, model, optimizer)
    model.train()
    model.to(tu.device)

    train(c, model, optimizer, sampler)
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



# if __name__ == '__main__':
#     main()