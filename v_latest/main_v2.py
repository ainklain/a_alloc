from copy import deepcopy
import argparse
import random
import re
import os
import torch
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt, cm
import GPUtil

# from ray import tune
# from ray.tune.schedulers import ASHAScheduler

from v20201222.logger_v2 import Logger
from v_latest.model_v2_bond import MyModel, load_model, save_model
from v20201222.dataset_v2 import DatasetManager, AplusData, MacroData, IncomeData, AssetData, DummyMacroData
from v20201222.optimizer_v2 import RAdam
import torch_utils as tu

# # #### profiler start ####
import builtins

try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func


    builtins.profile = profile


# # #### profiler end ####


def calc_y(wgt0, y1, cost_r=0.):
    # wgt0: 0 ~ T-1 ,  y1 : 1 ~ T  => 0 ~ T (0번째 값은 0)
    y = dict()
    wgt1 = wgt0 * (1 + y1)
    # turnover = np.append(np.sum(np.abs(wgt0[1:] - wgt1[:-1]), axis=1), 0)
    turnover = np.append(np.sum(np.abs(wgt0[1:] - wgt1[:-1] / wgt1[:-1].sum(axis=1, keepdims=True)), axis=1), 0)
    y['before_cost'] = np.insert(np.sum(wgt1, axis=1) - 1, 0, 0)
    y['after_cost'] = np.insert(np.sum(wgt1, axis=1) - 1 - turnover * cost_r, 0, 0)

    return y, turnover


class Trainer:
    def __init__(self, c, dataset_manager, tune_c=None):
        self.c = c
        self.tune_c = tune_c
        self.dataset_manager = dataset_manager
        self.loss_logger = Logger(self.__class__.__name__)

        self.reset_model()
        self.set_adaptive_configs(c.adaptive_flag)

    def reset_model(self):
        c = self.c
        if self.tune_c is not None:
            pre_lr = self.tunc_c['pre_lr']
            lr = self.tunc_c['lr']
        else:
            pre_lr = c.pre_lr
            lr = c.lr

        dm = self.dataset_manager
        self.model = MyModel(len(dm.features_list), len(dm.labels_list), configs=c)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=c.lr, weight_decay=0.01)
        self.pre_optimizer = RAdam(self.model.parameters(), lr=pre_lr, weight_decay=0.01)
        self.post_optimizer = RAdam(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.optimizer = None
        self.model.to(tu.device)

    def make_dir(self, t):
        c = self.c
        outpath_t = os.path.join(c.outpath, str(t))  # if os.path.isdir(outpath_t):
        r = re.compile("{}+".format(t))
        n = len(list(filter(r.match, os.listdir(c.outpath))))  # 동일 t 폴더 개수
        if n > 0:
            n2 = int(max(list(filter(r.match, os.listdir(c.outpath)))).split('_')[
                         -1]) + 1  # 동일 t폴더 중 가장 숫자 큰 수 (중간폴더 삭제시 개수꼬임)
        else:
            n2 = 0

        outpath_t = outpath_t + "_{}".format(max(n, n2))
        os.makedirs(outpath_t, exist_ok=True)

        str_ = c.export()
        with open(os.path.join(outpath_t, 'c.txt'), 'w') as f:
            f.write(str_)

        self.loss_logger.set_handler(c.log_level, 'loss_log', outpath=outpath_t, use_stream_handler=True)

        return outpath_t

    def check_earlystopping(self, losses_eval):
        if losses_eval > self.min_eval_loss:
            self.es_count += 1
        else:
            self.model.save_to_optim()
            self.min_eval_loss = losses_eval
            self.es_count = 0

        if self.max_count >= 0 and self.es_count >= self.max_count:
            return True
        else:
            return False

    def set_adaptive_configs(self, adaptive_flag):
        c = self.c
        self.min_eval_loss = float('inf')
        self.es_count = 0

        if adaptive_flag:
            self.max_count = c.adaptive_count
            # self.optimizer.param_groups[0]['lr'] = c.lr * c.adaptive_lrx
            self.optimizer = self.pre_optimizer
            self.losses_wgt_fixed = c.adaptive_loss_wgt
            self.use_n_batch_per_epoch = True
        else:
            self.max_count = c.es_max_count
            # self.optimizer.param_groups[0]['lr'] = c.lr
            self.optimizer = self.post_optimizer
            self.losses_wgt_fixed = c.loss_wgt
            self.use_n_batch_per_epoch = False

        return adaptive_flag

    def load_model(self, path):
        ep = load_model(path, self.model, self.optimizer)

    def run(self, t, use_plot=True):
        c = self.c
        outpath_t = self.make_dir(t)

        losses_train = losses_eval = losses_test = float('inf')
        losses_dict = {'train': np.zeros([c.num_epochs]),
                       'eval': np.zeros([c.num_epochs]),
                       'test': np.zeros([c.num_epochs])}

        adaptive_flag = self.set_adaptive_configs(c.adaptive_flag)
        for ep in range(c.num_epochs):
            if ep == c.num_epochs:
                self.model.load_from_optim()
                save_model(outpath_t, ep, self.model, self.optimizer)

            if ep % c.eval_freq == 0:
                losses_eval, early_stopped = self.eval(t, ep)
                # evaluate
                print('\n')
                self.loss_logger.info(
                    'ep: {} [es: {}/{} / min:{}] - train: {:.3f} / eval: {:.3f} / test: {:.3f}'.format(
                        ep, self.es_count, self.max_count, self.min_eval_loss, losses_train, losses_eval, losses_test))
                # GPUtil.showUtilization()
                if early_stopped and not adaptive_flag:
                    self.model.load_from_optim()
                    self.loss_logger.info('early stopped')
                    save_model(outpath_t, ep, self.model, self.optimizer)
                    break
                elif early_stopped and adaptive_flag:
                    # if adaptive_flag, reset to False and corresponding parameters
                    adaptive_flag = self.set_adaptive_configs(False)
                    # c.plot_freq = 1
                else:
                    assert not early_stopped

            if ep % c.plot_freq == 0:
                # test

                for mode in ['test_insample', 'test']:
                    is_insample = True if mode == 'test_insample' else False
                    losses_test, data_for_plot = self.test(t, is_insample=is_insample)

                    if use_plot:
                        suffix = "[tr={:.2f}][ev={:.2f}][te={:.2f}]".format(losses_train, losses_eval, losses_test)
                        date_dict = self.dataset_manager.get_begin_end_info(t, mode)
                        data_for_plot.update(date_dict)

                        self.plot(ep, data_for_plot, outpath_t, suffix=suffix + mode)

                losses_dict['train'][ep] = losses_train
                losses_dict['eval'][ep] = losses_eval
                losses_dict['test'][ep] = losses_test
                self.plot_learnig_curve(ep, outpath_t, losses_dict)

            losses_train = self.train(t)

        self.model.load_from_optim()
        if use_plot:
            for mode in ['test_insample', 'test']:
                is_insample = True if mode == 'test_insample' else False
                losses_test, data_for_plot = self.test(t, is_insample=is_insample)
                date_dict = self.dataset_manager.get_begin_end_info(t, mode)
                data_for_plot.update(date_dict)
                self.plot(20000, data_for_plot, outpath_t, suffix=mode)

    def train(self, t):
        self.model.train()
        c = self.c

        # get dataloader
        dataloader = self.dataset_manager.get_data_loader(t, 'train')

        # adaptive batching (when converge to guide then use epoch, else just sampled batch)
        if self.use_n_batch_per_epoch:
            n_batch_per_epoch = len(dataloader.dataset) // c.batch_size
        else:
            n_batch_per_epoch = 1

        losses_sum = 0
        for it_, data_dict in enumerate(dataloader):
            # assert to run one epoch
            if it_ > n_batch_per_epoch:
                break

            # use adversarial training (data augmentation)
            if c.adv_train is True:
                x = self.model.adversarial_noise(data_dict)

            # run model and training
            out = self.model.forward_with_loss(data_dict, losses_wgt_fixed=self.losses_wgt_fixed)
            if not out:
                continue
            _, losses, _, _, _ = out

            losses_np = tu.np_ify(losses)
            losses_sum += float(losses_np)

            self.optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.clip)
            # print([(n, x.grad) for n, x in list(self.model.named_parameters())])
            self.optimizer.step()

        return losses_sum

    def eval(self, t, ep):
        self.model.eval()
        c = self.c

        # get dataloader
        dataloader = self.dataset_manager.get_data_loader(t, 'eval')

        losses_sum = 0
        losses_dict = dict()
        # run model for all data to evaluate model

        with torch.set_grad_enabled(False):
            for data_dict in dataloader:
                out = self.model.forward_with_loss(data_dict, is_train=False, losses_wgt_fixed=self.losses_wgt_fixed)

                if not out:
                    continue

                _, losses, _, _, losses_dict = out

                losses_np = tu.np_ify(losses)
                losses_sum += float(losses_np)

        # check if earlystopped after pretrain (max_count can be 'adaptive' for adaptive-training)

        early_stopped = self.check_earlystopping(losses_sum) if ep >= c.n_pretrain * c.eval_freq else False

        print_str = "[eval] "
        for key, val in losses_dict.items():
            print_str += '{}:{:.3f} / '.format(key, tu.np_ify(val))
        self.loss_logger.info(print_str)

        return losses_sum, early_stopped

    def test(self, t, is_insample=False):
        self.model.eval()
        c = self.c

        # get dataloader
        mode = 'test_insample' if is_insample else 'test'
        dataloader = self.dataset_manager.get_data_loader(t, mode)
        losses_sum = 0

        # run model for all data to evaluate model
        with torch.set_grad_enabled(False):
            # store predicted wgts and t+1 returns of assets
            wgt = torch.zeros(len(dataloader.dataset), len(self.dataset_manager.labels_list)).to(tu.device)
            next_y = torch.zeros_like(wgt)
            for i, data_dict in enumerate(dataloader):
                out = self.model.forward_with_loss(data_dict, is_train=False, losses_wgt_fixed=self.losses_wgt_fixed)
                if not out:
                    continue
                wgt_i, losses, _, _, _ = out
                guide_wgt = self.model.random_guide_weight(c.base_weight, len(wgt), False).to(wgt_i.device)
                # assert len(wgt) == len(wgt_i)
                wgt[i * c.batch_size:(i + 1) * c.batch_size] = wgt_i[:]
                next_y[i * c.batch_size:(i + 1) * c.batch_size] = torch.exp(data_dict['labels']['logy']) - 1

                losses_np = tu.np_ify(losses)
                losses_sum += float(losses_np)

            data_for_plot = dict(wgt=tu.np_ify(wgt), guide_wgt=tu.np_ify(guide_wgt), next_y=tu.np_ify(next_y))
        return losses_sum, data_for_plot

    def plot(self, ep, data_for_plot: dict, outpath, suffix=''):
        c = self.c

        k_days = c.k_days
        cost_rate = c.cost_rate
        wgt_result = tu.np_ify(data_for_plot['wgt'])
        wgt_guide = tu.np_ify(data_for_plot['guide_wgt'])
        n_asset = wgt_result.shape[1]

        idx_list = [idx.split('_')[0] for idx in self.dataset_manager.labels_list]
        date_list = np.array(self.dataset_manager.dataset.idx)
        date_ = date_list[(date_list >= data_for_plot['date_'][0]) & (date_list <= data_for_plot['date_'][-1])]

        if len(data_for_plot['idx_']) == 4:
            # test_insample: [train_begin, train_end, test_begin, test_end]
            # test_insample에서 test시점의 날짜 기준으로 맞추는 작업
            selected_sampling = np.arange(data_for_plot['idx_'][2] % k_days, len(wgt_result), k_days)
        else:
            # test: [test_begin, test_end]
            selected_sampling = np.arange(0, len(wgt_result), k_days)

        y_next = data_for_plot['next_y'][selected_sampling, :]
        wgt_result_calc = wgt_result[selected_sampling, :]
        wgt_guide_calc = wgt_guide[selected_sampling, :]

        # min weight constraint (bigger than half of guide weight)
        is_bigger_than_half = (wgt_result_calc >= wgt_guide_calc / 2)
        const_multiplier = (1 - (wgt_guide_calc / 2 * ~is_bigger_than_half).sum(axis=1, keepdims=True)) / (
                wgt_result_calc * is_bigger_than_half).sum(axis=1, keepdims=True)
        wgt_result_const_calc = wgt_result_calc * is_bigger_than_half * const_multiplier + wgt_guide_calc / 2 * ~is_bigger_than_half

        # features, labels = test_features, test_labels; wgt = wgt_result
        # active_share = np.sum(np.abs(wgt_result - wgt_base), axis=1)
        active_share = np.sum(np.abs(wgt_result_calc - wgt_guide_calc), axis=1)

        y_guide = np.insert(np.sum((1 + y_next) * wgt_guide_calc, axis=1), 0, 1.)
        y_port = np.insert(np.sum((1 + y_next) * wgt_result_calc, axis=1), 0, 1.)
        y_port_const = np.insert(np.sum((1 + y_next) * wgt_result_const_calc, axis=1), 0, 1.)
        y_eq = np.insert(np.mean(1 + y_next, axis=1), 0, 1.)

        y_port_with_c, turnover_port = calc_y(wgt_result_calc, y_next, cost_rate)
        y_port_const_with_c, turnover_port_const = calc_y(wgt_result_const_calc, y_next, cost_rate)
        y_guide_with_c, turnover_guide = calc_y(wgt_guide_calc, y_next, cost_rate)

        x = np.arange(len(y_guide))

        # save data
        df_wgt = pd.DataFrame(data=wgt_result, index=date_, columns=[idx_nm + '_wgt' for idx_nm in idx_list])

        date_test_selected = ((date_[selected_sampling] >= data_for_plot['date_'][-2]) & (
                    date_[selected_sampling] < data_for_plot['date_'][-1]))
        date_selected = date_[selected_sampling]

        columns = [idx_nm + '_wgt' for idx_nm in idx_list] + [idx_nm + '_wgt_const' for idx_nm in idx_list] + [
            idx_nm + '_ynext' for idx_nm in idx_list] + ['port_bc', 'port_const_bc', 'guide_bc', 'port_ac',
                                                         'port_const_ac', 'guide_ac']
        df = pd.DataFrame(data=np.concatenate([wgt_result_calc, wgt_result_const_calc, y_next,
                                               y_port_with_c['before_cost'][1:, np.newaxis],
                                               y_port_const_with_c['before_cost'][1:, np.newaxis],
                                               y_guide_with_c['before_cost'][1:, np.newaxis],
                                               y_port_with_c['after_cost'][1:, np.newaxis],
                                               y_port_const_with_c['after_cost'][1:, np.newaxis],
                                               y_guide_with_c['after_cost'][1:, np.newaxis],
                                               ], axis=-1),
                          index=date_selected
                          , columns=columns)

        df_all = df.loc[:, ['port_bc', 'port_const_bc', 'guide_bc', 'port_ac', 'port_const_ac', 'guide_ac']]
        df_test = df.loc[
            date_test_selected, ['port_bc', 'port_const_bc', 'guide_bc', 'port_ac', 'port_const_ac', 'guide_ac']]
        df_stats = pd.concat({'mu_all': df_all.mean() * 12,
                              'sig_all': df_all.std(ddof=1) * np.sqrt(12),
                              'sr_all': df_all.mean() / df_all.std(ddof=1) * np.sqrt(12),
                              'mu_test': df_test.mean() * 12,
                              'sig_test': df_test.std(ddof=1) * np.sqrt(12),
                              'sr_test': df_test.mean() / df_test.std(ddof=1) * np.sqrt(12)},
                             axis=1)

        print(ep, suffix, '\n', df_stats)
        df.to_csv(os.path.join(outpath, '{}_all_data_{}.csv'.format(ep, suffix)))
        df_stats.to_csv(os.path.join(outpath, '{}_stats_{}.csv'.format(ep, suffix)))
        df_wgt.to_csv(os.path.join(outpath, '{}_wgtdaily_{}.csv'.format(ep, suffix)))

        # ################ together

        outpath_plot = os.path.join(outpath, 'plot')
        os.makedirs(outpath_plot, exist_ok=True)

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        # ax = plt.gca()
        l_port, = ax1.plot(x, y_port.cumprod())
        l_port_const, = ax1.plot(x, y_port_const.cumprod())
        l_eq, = ax1.plot(x, y_eq.cumprod())
        l_guide, = ax1.plot(x, y_guide.cumprod())
        ax1.legend(handles=(l_port, l_port_const, l_eq, l_guide), labels=('port', 'port_const', 'eq', 'guide'))

        ax2 = fig.add_subplot(212)
        l_port_guide, = ax2.plot(x, (1 + y_port - y_guide).cumprod() - 1.)
        l_portconst_guide, = ax2.plot(x, (1 + y_port_const - y_guide).cumprod() - 1.)
        # l_port_eq, = ax2.plot(x,(1 + y_port - y_eq).cumprod() - 1.)
        l_port_guide_ac, = ax2.plot(x, (1 + y_port_with_c['after_cost'] - y_guide_with_c['after_cost']).cumprod() - 1.)
        l_portconst_guide_ac, = ax2.plot(x, (
                1 + y_port_const_with_c['after_cost'] - y_guide_with_c['after_cost']).cumprod() - 1.)
        ax2.legend(handles=(l_port_guide, l_portconst_guide, l_port_guide_ac, l_portconst_guide_ac),
                   labels=('port-guide', 'portconst-guide', 'port-guide(ac)', 'portconst-guide(ac)'))
        # ax2.legend(handles=(l_port_guide, l_portconst_guide, l_port_eq, l_port_guide_ac, l_portconst_guide_ac),
        #            labels=('port-guide', 'portconst-guide', 'port-eq','port-guide(ac)', 'portconst-guide(ac)'))

        if len(data_for_plot['idx_']) == 4:
            ax1.axvline(data_for_plot['idx_'][1] // k_days)
            ax1.axvline(data_for_plot['idx_'][2] // k_days)
            ax2.axvline(data_for_plot['idx_'][1] // k_days)
            ax2.axvline(data_for_plot['idx_'][2] // k_days)

            ax1.text(x[0], ax1.get_ylim()[1], data_for_plot['date_'][0]
                     , horizontalalignment='center'
                     , verticalalignment='center'
                     , bbox=dict(facecolor='white', alpha=0.7))
            ax1.text(data_for_plot['idx_'][1] // k_days, ax1.get_ylim()[1], data_for_plot['date_'][1]
                     , horizontalalignment='center'
                     , verticalalignment='center'
                     , bbox=dict(facecolor='white', alpha=0.7))
            ax2.text(data_for_plot['idx_'][2] // k_days, ax2.get_ylim()[1], data_for_plot['date_'][2]
                     , horizontalalignment='center'
                     , verticalalignment='center'
                     , bbox=dict(facecolor='white', alpha=0.7))
            ax2.text(x[-1], ax2.get_ylim()[1], data_for_plot['date_'][3]
                     , horizontalalignment='center'
                     , verticalalignment='center'
                     , bbox=dict(facecolor='white', alpha=0.7))
        else:
            ax1.text(x[0], ax1.get_ylim()[1], data_for_plot['date_'][0]
                     , horizontalalignment='center'
                     , verticalalignment='center'
                     , bbox=dict(facecolor='white', alpha=0.7))
            ax1.text(x[-1], ax1.get_ylim()[1], data_for_plot['date_'][1]
                     , horizontalalignment='center'
                     , verticalalignment='center'
                     , bbox=dict(facecolor='white', alpha=0.7))

        fig.savefig(os.path.join(outpath_plot, '{}_test_y_{}.png'.format(ep, suffix)))
        plt.close(fig)

        # #############################

        viridis = cm.get_cmap('viridis', n_asset)

        x = np.arange(len(wgt_result_calc))
        wgt_result_cum = wgt_result_calc.cumsum(axis=1)
        wgt_guide_cum = wgt_guide_calc.cumsum(axis=1)
        fig = plt.figure()
        fig.suptitle('Weight Diff')
        # ax1 = fig.add_subplot(311)
        ax1 = fig.add_subplot(211)
        ax1.set_title('base')
        for i in range(n_asset):
            if i == 0:
                ax1.fill_between(x, 0, wgt_result_cum[:, i], facecolor=viridis.colors[i], alpha=.7)
            else:
                ax1.fill_between(x, wgt_result_cum[:, i - 1], wgt_result_cum[:, i], facecolor=viridis.colors[i],
                                 alpha=.7)

        # ax2 = fig.add_subplot(312)
        ax2 = fig.add_subplot(212)
        ax2.set_title('result')
        for i in range(n_asset):
            if i == 0:
                ax2.fill_between(x, 0, wgt_guide_cum[:, i], facecolor=viridis.colors[i], alpha=.7)
            else:
                ax2.fill_between(x, wgt_guide_cum[:, i - 1], wgt_guide_cum[:, i], facecolor=viridis.colors[i],
                                 alpha=.7)

        if len(data_for_plot['idx_']) == 4:
            ax1.axvline(data_for_plot['idx_'][1] // k_days)
            ax1.axvline(data_for_plot['idx_'][2] // k_days)

            ax2.axvline(data_for_plot['idx_'][1] // k_days)
            ax2.axvline(data_for_plot['idx_'][2] // k_days)

            ax1.text(x[0], ax1.get_ylim()[1], data_for_plot['date_'][0]
                     , horizontalalignment='center'
                     , verticalalignment='center'
                     , bbox=dict(facecolor='white', alpha=0.7))
            ax1.text(data_for_plot['idx_'][1] // k_days, ax1.get_ylim()[1], data_for_plot['date_'][1]
                     , horizontalalignment='center'
                     , verticalalignment='center'
                     , bbox=dict(facecolor='white', alpha=0.7))
            ax2.text(data_for_plot['idx_'][2] // k_days, ax1.get_ylim()[1], data_for_plot['date_'][2]
                     , horizontalalignment='center'
                     , verticalalignment='center'
                     , bbox=dict(facecolor='white', alpha=0.7))
            ax2.text(x[-1], ax1.get_ylim()[1], data_for_plot['date_'][3]
                     , horizontalalignment='center'
                     , verticalalignment='center'
                     , bbox=dict(facecolor='white', alpha=0.7))

        fig.savefig(os.path.join(outpath_plot, '{}_test_wgt_{}.png'.format(ep, suffix)))
        plt.close(fig)

    def plot_learnig_curve(self, ep, outpath, losses_dict):
        c = self.c
        ##############################
        x_plot = np.arange(c.num_epochs)
        x_len = (ep // 10000 + 1) * 10000
        sampling_freq = (ep // 10000 + 1) * 1000
        fig = plt.figure()
        l_train, = plt.plot(x_plot[1:x_len], losses_dict['train'][1:x_len])
        l_eval, = plt.plot(x_plot[1:x_len], losses_dict['eval'][1:x_len])
        loss_test = np.zeros_like(losses_dict['test'][:x_len])
        loss_test[::sampling_freq] = losses_dict['test'][:x_len][::sampling_freq]
        l_test, = plt.plot(x_plot[1:x_len], loss_test[1:])

        plt.legend(handles=(l_train, l_eval, l_test), labels=('train', 'eval', 'test'))

        fig.savefig(os.path.join(outpath, 'learning_curve.png'))
        plt.close(fig)

    def plot_macro(self, outpath):
        _, m_data = self.dataset_manager._original_recent_250d
        for c in m_data.columns:
            fig = plt.figure()
            plt.plot(m_data[c])
            plt.xticks(np.arange(0, len(m_data), 50), m_data.index[::50])
            plt.title(c)
            fig.savefig(os.path.join(outpath, '{}.png'.format(c)))
            plt.close(fig)

    def run_all(self):
        c = self.c
        iter_ = range(c.base_i0, self.dataset_manager.max_len, c.retrain_days)
        for t in iter_:
            self.reset_model()
            self.run(t)


def income():
    base_weight = dict(h=[0.8, 0.2],
                       m=[0.6, 0.4],
                       l=[0.45, 0.55],
                       eq=[0.5, 0.5])

    for seed, suffix in zip([100, 1000, 123], ["_0", "_1", "_2"]):
        for key in ['l', 'm', 'h', 'eq', ]:
            # for seed, suffix in zip([100, 1000], ["_0", "_1"]):
            #     for key in ['m']:
            # configs & variables
            name = 'income01_k20_{}'.format(key)
            # name = 'app_adv_1'
            c = Configs(name)
            c.base_weight = base_weight[key]
            c.seed = seed
            c.cash_idx = 1
            # seed
            random.seed(c.seed)
            np.random.seed(c.seed)
            torch.manual_seed(c.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            str_ = c.export()
            with open(os.path.join(c.outpath, 'c.txt'), 'w') as f:
                f.write(str_)

            # data processing

            data_list = [IncomeData(), MacroData()]
            dm = DatasetManager(data_list, c.test_days, c.batch_size)

            trainer = Trainer(c, dm)
            trainer.run_all()


testmode = False


@profile
def main(testmode=False, args=None):
    base_weight = dict(h=[0.69, 0.2, 0.1, 0.01],
                       m=[0.4, 0.1, 0.075, 0.425],
                       l=[0.25, 0.05, 0.05, 0.65],
                       eq=[0.25, 0.25, 0.25, 0.25])
    # # 실제
    # base_weight = dict(h=[0.69, 0.2, 0.1, 0.01],
    #                    m=[0.45, 0.15, 0.075, 0.325],
    #                    l=[0.30, 0.1, 0.05, 0.55],
    #                    eq=[0.25, 0.25, 0.25, 0.25])

    # 검증
    # base_weight = dict(h=[0.69, 0.2, 0.1, 0.01],
    #                    m=[0.4, 0.15, 0.075, 0.375],
    #                    l=[0.25, 0.1, 0.05, 0.6],
    #                    eq=[0.25, 0.25, 0.25, 0.25])

    # for seed, suffix in zip([100, 1000, 123], ["_0", "_1", "_2"]):
    # for seed, suffix in zip([1000, 123], ["_0", "_1", "_2"]):
    #     for key in ['m','l','h','eq',]:
    # for seed, suffix in zip([100, 1000], ["_0", "_1"]):
    for seed, suffix in zip([123], ["_0"]):
        # for aa in [# ['spx500','bbusagtr'],
        #             ['gscigold','bbusagtr'],
        # ['spx500', 'gscigold', 'bbusagtr'],
        # ['spx500', 'russell2000', 'gscigold', 'bbusagtr'],
        # ['spx500', 'nasdaq100', 'russell2000', 'gscigold', 'bbusagtr'],
        # ['spx500', 'russell2000', 'kospi200', 'gscigold', 'bbusagtr', 'kiscompbondcall'],
        # ['spx500', 'russell2000', 'csi300', 'kospi200', 'gscigold', 'bbusagtr', 'kiscompbondcall'],]:
        for key in ['l', 'm', 'h', 'eq']:
            # m_data_raw = MacroData('macro_data_20201222.txt')
            # for m in m_data_raw.columns:
            #     m_data = MacroData('macro_data_20201222.txt')
            # m = ['usyc3m2y index','gdp cury index','cl1 comdty','indu index']
            # m_data.df = m_data.df.loc[:, m]
            # m_data.columns = m
            aa = ''
            aa_str = '-'.join(aa)
            # key = 'eq'
            lr = 0.1
            # configs & variables
            if args is not None:
                name = '{}_{}'.format(args.prefix, key + str(lr) + aa_str)
            else:
                name = '{}'.format(key + str(lr) + aa_str)

            # name = 'app_adv_1'
            c = Configs(name)
            c.base_weight = base_weight[key]
            c.seed = seed

            if key == 'h':
                c.loss_wgt['wgt_guide'] = 0.02
                c.loss_wgt['logy'] = 0.1
                # c.wgt_range = 0.11 * 2
                c.wgt_range_min = [0.5, 0.1, 0., 0.01]
                c.wgt_range_max = [1., 1., 0.3, 0.3]

                c.mdd_cp = 0.1
                c.lr = 2e-2
                model_path = './out/test_20201222_01_h/3900_0'

                # l: ./out/test_20201222_01_l/3900_0
                # m: test_20201222_01_m
                # h: test_20201222_01_h
                # eq: test_20201222_01_eq
            elif key == 'm':
                c.loss_wgt['wgt_guide'] = 0.02
                # c.wgt_range = 0.08 * 2
                c.wgt_range_min = [0.2, 0.05, 0.01, 0.4]
                c.wgt_range_max = [0.65, 0.65, 0.15, 0.6]

                c.mdd_cp = 0.05
                model_path = './out/test_20201222_01_m/3900_0'
            elif key == 'l':
                c.loss_wgt['wgt_guide'] = 0.02
                # c.wgt_range = 0.24 * 2
                c.wgt_range_min = [0.1, 0.02, 0., 0.6]
                c.wgt_range_max = [0.45, 0.45, 0.1, 0.8]
                c.mdd_cp = 0.03
                model_path = './out/test_20201222_01_l/3900_0'
            else:
                # c.loss_wgt['wgt_guide'] = 0.0
                # c.wgt_range = 0.99
                c.lr = 5e-2
                c.wgt_range_min = [0., 0., 0., 0.]
                c.wgt_range_max = [1., 1., 1., 1.]
                c.mdd_cp = 0.05
                model_path = './out/test_20201222_01_eq/3900_0'

            # seed
            random.seed(c.seed)
            np.random.seed(c.seed)
            torch.manual_seed(c.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            select = False
            if select is True:
                c.loss_wgt['mdd_pf'] = 0.1
                c.lr = lr

                a_data = AssetData('asset_data_us_ext_20201222.txt')
                from collections import OrderedDict

                label_columns_dict = OrderedDict()
                for feature in ['logy', 'mu', 'sigma']:
                    label_columns_dict[feature] = ['{}_{}'.format(col, feature) for col in aa]

                a_data.df = a_data.df.loc[:, aa]
                a_data.columns = aa
                a_data.label_columns_dict = label_columns_dict

                data_list = [MacroData('macro_data_20201222.txt'), a_data]
                c.cash_idx = len(aa) - 1
                c.base_weight = [1. / len(aa)] * len(aa)
                c.wgt_range_min = [0.] * len(aa)
                c.wgt_range_max = [1.] * len(aa)
                ii = 3500
            else:
                c.cash_idx = 3
                # data processing
                if c.cash_idx == 2:
                    # data_list = [m_data, AssetData('asset_data_us_20201222.txt')]
                    data_list = [MacroData(), AssetData('asset_data_us_20201222.txt')]
                    # data_list = [MacroData('dummy_macro_data_20201222.txt'), AssetData('asset_data_us_20201222.txt')]
                    c.loss_wgt['mdd_pf'] = 0.1
                    c.lr = lr

                    ii = 3500
                elif c.cash_idx == 3:
                    data_list = [AplusData('app_data_20201230.txt'), MacroData('macro_data_20210104.txt')]
                    ii = 3900
                elif c.cash_idx == 7:
                    data_list = [AssetData('asset_data_kor_ext_20201222.txt'), MacroData()]
                    ii = 3500
                    c.base_weight = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
                    c.wgt_range_min = [0., 0., 0., 0., 0., 0., 0., 0.]
                    c.wgt_range_max = [1., 1., 1., 1., 1., 1., 1., 1.]
                    c.loss_wgt['logy'] = 0.1

                else:
                    raise NotImplementedError

            c.load(model_path)
            str_ = c.export()
            with open(os.path.join(c.outpath, 'c.txt'), 'w') as f:
                f.write(str_)

            # data_list = [Index(), MacroData()]
            dm = DatasetManager(data_list, c.test_days, c.batch_size)

            trainer = Trainer(c, dm)
            # trainer.run_all()
            trainer.run(ii)
            # train(c, model, optimizer, dm, 3489)
            # train(c, model, optimizer, sampler, t=1700)

            # backtest(c, sampler, suffix)


def load_and_run(key, model_path):
    from ast import literal_eval

    base_weight = dict(h=[0.69, 0.2, 0.1, 0.01],
                       m=[0.4, 0.1, 0.075, 0.425],
                       l=[0.25, 0.05, 0.05, 0.65],
                       eq=[0.25, 0.25, 0.25, 0.25])

    # key = 'eq'
    # model_path = './out/!test1_eq/3900_0'
    # c_load = pd.read_csv(os.path.join(model_path, 'c.txt'))

    # load and parse 'c.txt'
    p = re.compile('[a-zA-Z0-9_]+[:]')
    with open(os.path.join(model_path, 'c.txt'), 'r') as f:
        texts = ''.join(f.readlines())

    tags = p.findall(texts)

    c = Configs('')
    for tag in tags[::-1]:
        print(tag)
        *texts, val = texts.split(tag)
        texts = tag.join(texts)
        print(val)

        a = tag.split(':')[0]
        val = val.strip()
        if val in ['True', 'False']:
            c.__setattr__(a, bool(val))
        elif re.match("\[.+\]", val) is not None or re.match("\{.+\}", val) is not None:
            c.__setattr__(a, literal_eval(val))
        elif re.fullmatch("[0-9.]+", val):
            if "." in val:
                c.__setattr__(a, float(val))
            else:
                c.__setattr__(a, int(val))
        else:
            c.__setattr__(a, val)

    # seed
    tu.set_seed(c.seed)

    data_list = [AplusData('app_data_20201230.txt'), MacroData('macro_data_20210104.txt')]
    dm = DatasetManager(data_list, c.test_days, c.batch_size)
    trainer = Trainer(c, dm)
    trainer.load_model(model_path)

    for is_insample in [True, False]:
        t = 3900
        # is_insample = True
        mode = 'test_insample' if is_insample else 'test'
        losses_test, data_for_plot = trainer.test(t, is_insample=is_insample)
        date_dict = trainer.dataset_manager.get_begin_end_info(t, mode)
        data_for_plot.update(date_dict)
        trainer.plot(30000, data_for_plot, model_path, suffix=mode)
        print(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='test', type=str)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--base_key', default='eq', type=str)
    parser.add_argument('--model_path', default=None, type=str)

    args = parser.parse_args()
    if not args.test:
        main(args=args)
    else:
        assert args.model_path is not None
        load_and_run(args.base_key, args.model_path)

        # l: ./out/test_20201222_01_l/3900_0
        # m: test_20201222_01_m
        # h: test_20201222_01_h
        # eq: test_20201222_01_eq
    # income()

