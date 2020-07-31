from copy import deepcopy
import random
import re
import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt, cm
import GPUtil


from logger_v2 import Logger
from model_v2 import MyModel, load_model, save_model
from dataset_v2 import DatasetManager, AplusData, MacroData
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
        self.comment = """
        wgt_mu = 0.5 + 0.99 * torch.tanh(wgt_mu) + 1e-6
        wgt_sigma = 0.2 * F.softplus(wgt_logsigma) + 1e-6
        """

        self.seed = 1000
        self.name = name
        self.lr = 1e-3
        self.batch_size = 256
        self.num_epochs = 1000
        self.base_i0 = 2000
        self.mc_samples = 200
        self.sampling_freq = 20
        self.k_days = 20
        self.label_days = 20
        self.strategy_days = 250

        # adaptive / earlystopping
        self.adaptive_flag = True
        self.adaptive_count = 5
        self.adaptive_lrx = 0.5  # learning rate * 배수
        self.es_max_count = 20
        self.adaptive_loss_threshold = None  # -1

        self.retrain_days = 240
        self.test_days = 2000  # test days
        self.init_train_len = 500
        self.train_data_len = 2000
        self.normalizing_window = 500  # norm windows for macro data
        self.use_accum_data = True  # [sampler] 데이터 누적할지 말지
        self.adv_train = True
        self.n_pretrain = 5
        self.max_entropy = True

        self.loss_threshold = None  # -1

        self.datatype = 'app'
        # self.datatype = 'inv'

        self.cost_rate = 0.003
        self.plot_freq = 50
        self.eval_freq = 1 # 20
        self.save_freq = 20
        self.model_init_everytime = False
        self.use_guide_wgt_as_prev_x = False  # model / forward_with_loss

        self.hidden_dim = [72, 48, 32]
        self.alloc_hidden_dim = [32, 32]
        self.dropout_r = 0.3

        self.random_guide_weight = 0.1
        self.random_flip = 0.1  # flip sign
        self.random_label = 0.1  # random sample from dist.

        self.clip = 1.

        # logger
        self.log_level = 'DEBUG'

        ## attention
        self.d_model = 64
        self.n_heads = 4
        self.d_k = self.d_v = self.d_model // self.n_heads
        self.d_ff = 128

        ## FiLM
        self.use_condition_network = True
        self.film_hidden_dim = [16, 16]

        # self.loss_wgt = {'y_pf': 1., 'mdd_pf': 1., 'logy': 1., 'wgt': 0., 'wgt2': 0.01, 'wgt_guide': 0., 'cost': 0., 'entropy': 0.}
        self.loss_list = ['y_pf', 'mdd_pf', 'logy', 'wgt_guide', 'cost', 'entropy']
        self.adaptive_loss_wgt = {'y_pf': 0., 'mdd_pf': 0., 'logy': 1., 'wgt_guide': 0.1, 'cost': 0., 'entropy': 0.0}
        self.loss_wgt = {'y_pf': 1, 'mdd_pf': 1000., 'logy': 1., 'wgt_guide': 0.02, 'cost': 1., 'entropy': 0.001}
        # self.adaptive_loss_wgt = {'y_pf': 1, 'mdd_pf': 1000., 'logy': 1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.01, 'cost': 1., 'entropy': 0.001}

        # default
        # self.adaptive_loss_wgt = {'y_pf': 1, 'mdd_pf': 1000., 'logy': 1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.02, 'cost': 1., 'entropy': 0.001}
        # good balance
        # self.adaptive_loss_wgt = {'y_pf': 1, 'mdd_pf': 1000., 'logy': 1., 'wgt': 0., 'wgt2': 0., 'wgt_guide': 0.0002,
        #                       'cost': 2., 'entropy': 0.0001}
        self.init()
        self.set_path()

    def init(self):
        self.init_weight()

    def init_weight(self):
        if self.datatype == 'app':
            self.cash_idx = 3
            self.base_weight = [0.25, 0.1, 0.05, 0.6]
        else:
            self.cash_idx = 0
            self.base_weight = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
            # self.base_weight = None

    def set_path(self):
        self.outpath = './out/{}/'.format(self.name)
        os.makedirs(self.outpath, exist_ok=True)

    def export(self):
        return_str = ""
        for key in self.__dict__.keys():
            return_str += "{}: {}\n".format(key, self.__dict__[key])

        return return_str


def calc_y(wgt0, y1, cost_r=0.):
    # wgt0: 0 ~ T-1 ,  y1 : 1 ~ T  => 0 ~ T (0번째 값은 0)
    y = dict()
    wgt1 = wgt0 * (1 + y1)
    # turnover = np.append(np.sum(np.abs(wgt0[1:] - wgt1[:-1]), axis=1), 0)
    turnover = np.append(np.sum(np.abs(wgt0[1:] - wgt1[:-1]/wgt1[:-1].sum(axis=1, keepdims=True)), axis=1), 0)
    y['before_cost'] = np.insert(np.sum(wgt1, axis=1) - 1, 0, 0)
    y['after_cost'] = np.insert(np.sum(wgt1, axis=1) - 1 - turnover * cost_r, 0, 0)

    return y, turnover


class Trainer:
    def __init__(self, c, dataset_manager):
        self.c = c
        self.dataset_manager = dataset_manager
        self.loss_logger = Logger(self.__class__.__name__)

        self.reset_model()
        self.set_adaptive_configs(c.adaptive_flag)

    def reset_model(self):
        c = self.c
        dm = self.dataset_manager
        self.model = MyModel(len(dm.features_list), len(dm.labels_list), configs=c)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=c.lr, weight_decay=0.01)

        self.model.to(tu.device)

    def make_dir(self, t):
        c = self.c
        outpath_t = os.path.join(c.outpath, str(t))     # if os.path.isdir(outpath_t):
        r = re.compile("{}+".format(t))
        n = len(list(filter(r.match, os.listdir(c.outpath))))   # 동일 t 폴더 개수
        if n > 0:
            n2 = int(max(list(filter(r.match, os.listdir(c.outpath)))).split('_')[-1]) + 1 # 동일 t폴더 중 가장 숫자 큰 수 (중간폴더 삭제시 개수꼬임)
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
            self.optimizer.param_groups[0]['lr'] = c.lr * c.adaptive_lrx
            self.losses_wgt_fixed = c.adaptive_loss_wgt
        else:
            self.max_count = c.es_max_count
            self.optimizer.param_groups[0]['lr'] = c.lr
            self.losses_wgt_fixed = c.loss_wgt

        return adaptive_flag

    def run(self, t, use_plot=True):
        c = self.c
        outpath_t = self.make_dir(t)

        losses_train = losses_eval = losses_test = float('inf')
        losses_dict = {'train': np.zeros([c.num_epochs]), 'eval': np.zeros([c.num_epochs]), 'test': np.zeros([c.num_epochs])}

        adaptive_flag = self.set_adaptive_configs(c.adaptive_flag)
        for ep in range(c.num_epochs):
            if ep % c.eval_freq == 0:
                losses_eval, early_stopped = self.eval(t, ep)
                # evaluate
                print('\n')
                self.loss_logger.info('ep: {} [es: {}/{} / min:{}] - train: {:.3f} / eval: {:.3f} / test: {:.3f}'.format(
                    ep, self.es_count, self.max_count, self.min_eval_loss, losses_train, losses_eval, losses_test))
                # GPUtil.showUtilization()
                if early_stopped and not adaptive_flag:
                    self.model.load_from_optim()
                    self.loss_logger.info('early stopped')
                    break
                elif early_stopped and adaptive_flag:
                    # if adaptive_flag, reset to False and corresponding parameters
                    adaptive_flag = self.set_adaptive_configs(False)
                else:
                    assert not early_stopped

            if ep % c.plot_freq == 0:
                # test
                losses_test, data_for_plot = self.test(t, is_insample=True)

                if use_plot:

                    suffix = "[tr={:.2f}][ev={:.2f}][te={:.2f}]".format(losses_train, losses_eval, losses_test)

                    date_dict = self.dataset_manager.get_begin_end_info(t, 'test_insample')
                    data_for_plot.update(date_dict)
                    self.plot(ep, data_for_plot, outpath_t, suffix=suffix)

                    losses_dict['train'][ep] = losses_train
                    losses_dict['eval'][ep] = losses_eval
                    losses_dict['test'][ep] = losses_test
                    self.plot_learnig_curve(ep, outpath_t, losses_dict)

            losses_train = self.train(t)

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
        n_batch_per_epoch = len(dataloader.dataset) // c.batch_size

        losses_sum = 0
        for it_, data_dict in enumerate(dataloader):
            # assert to run one epoch
            if it_ > n_batch_per_epoch:
                break

            # use adversarial training (data augmentation)
            if c.adv_train is True:
                x = self.model.adversarial_noise(data_dict)

            # run model and training
            _, losses, _, _, losses_dict = self.model.forward_with_loss(data_dict, losses_wgt_fixed=self.losses_wgt_fixed)

            losses_np = tu.np_ify(losses)
            losses_sum += float(losses_np)

            self.optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.clip)
            self.optimizer.step()

        return losses_sum

    def eval(self, t, ep):
        self.model.eval()
        c = self.c

        # get dataloader
        dataloader = self.dataset_manager.get_data_loader(t, 'eval')

        losses_sum = 0

        # run model for all data to evaluate model
        with torch.set_grad_enabled(False):
            for data_dict in dataloader:
                _, losses, _, _, losses_dict = self.model.forward_with_loss(data_dict, is_train=False, losses_wgt_fixed=self.losses_wgt_fixed)

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
                wgt_i, losses, _, _, _ = self.model.forward_with_loss(data_dict, is_train=False, losses_wgt_fixed=self.losses_wgt_fixed)
                guide_wgt = self.model.random_guide_weight(c.base_weight, len(wgt), False).to(wgt_i.device)
                assert len(wgt) == len(wgt_i)
                wgt[:] = wgt_i[:]
                next_y[:] = torch.exp(data_dict['labels']['logy'])-1

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
        date_ = date_list[(date_list >= data_for_plot['date_'][0]) & (date_list < data_for_plot['date_'][-1])]

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

        date_test_selected = ((date_[selected_sampling] >= data_for_plot['date_'][-2]) & (date_[selected_sampling] < data_for_plot['date_'][-1]))
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

        fig.savefig(os.path.join(outpath, '{}_test_y_{}.png'.format(ep, suffix)))
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
                ax1.fill_between(x, wgt_result_cum[:, i - 1], wgt_result_cum[:, i], facecolor=viridis.colors[i], alpha=.7)

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

        fig.savefig(os.path.join(outpath, '{}_test_wgt_{}.png'.format(ep, suffix)))
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

    def run_all(self):
        c = self.c
        iter_ = range(c.base_i0, self.dataset_manager.max_len, c.retrain_days)
        for t in iter_:
            self.reset_model()
            self.run(t)


testmode = False
@profile
def main(testmode=False):

    # 실제
    base_weight = dict(h=[0.69, 0.2, 0.1, 0.01],
                       m=[0.45, 0.15, 0.075, 0.325],
                       l=[0.30, 0.1, 0.05, 0.55],
                       eq=[0.25, 0.25, 0.25, 0.25])
    # 검증
    # base_weight = dict(h=[0.69, 0.2, 0.1, 0.01],
    #                    m=[0.4, 0.15, 0.075, 0.375],
    #                    l=[0.25, 0.1, 0.05, 0.6],
    #                    eq=[0.25, 0.25, 0.25, 0.25])

    # for seed, suffix in zip([100, 1000, 123], ["_0", "_1", "_2"]):
        # for key in ['m','l','h','eq',]:
    for seed, suffix in zip([100], ["_0"]):
        for key in ['l', 'm', 'h', 'eq']:
            # configs & variables
            name = 'attn03_{}'.format(key)
            # name = 'app_adv_1'
            c = Configs(name)
            c.base_weight = base_weight[key]
            c.seed = seed
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

            data_list = [AplusData(), MacroData()]
            dm = DatasetManager(data_list, c.test_days, c.batch_size)

            trainer = Trainer(c, dm)
            trainer.run_all()
            # trainer.run(3489)
            # train(c, model, optimizer, dm, 3489)
            # train(c, model, optimizer, sampler, t=1700)

            # backtest(c, sampler, suffix)


if __name__ == '__main__':
    main()