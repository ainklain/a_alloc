# renewed from model_attn

import copy
from collections import OrderedDict
import os
import numpy as np
from typing import List, Union, Any
import torch
from torch import nn, distributions
from torch.nn import init, Module, functional as F
from torch.utils.data import DataLoader
from torch.distributions import Normal
import pytorch_lightning as pl
from torch.distributions import Categorical


import layer
import torch_utils as tu
from v_latest.optimizer_v2 import RAdam

# ##### for using profiler without error ####
tu.use_profile()


class MomentProp(Module):
    def __init__(self,
                 in_dim: int = None,
                 out_dim: int = None,
                 hidden_dim: List[int] = None,
                 p_drop: float = 0.3):
        super(MomentProp, self).__init__()

        self.p_drop = p_drop
        self.hidden_layers = nn.ModuleList()

        h_in = in_dim
        for h_out in hidden_dim:
            self.hidden_layers.append(nn.Linear(h_in, h_out))
            h_in = h_out

        self.out_layer = nn.Linear(h_out, out_dim)

    def forward(self,
                x,
                sample=True):
        """
        x = torch.randn(200, 512, 30).cuda()
        """
        mask = self.training or sample
        for h_layer in self.hidden_layers:
            x = h_layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.p_drop, training=self.training) * (1 - self.p_drop)
            # x = F.dropout(x, p=self.p_drop, training=mask)

        x = self.out_layer(x)
        return x

    def predict(self, x):
        p_drop = self.p_drop
        pred_mu = x.detach().clone()
        pred_var = torch.zeros_like(x, requires_grad=False)
        with torch.no_grad():

            for h_layer in self.hidden_layers:

                # linear
                pred_mu = h_layer(pred_mu)
                pred_var = torch.mm(pred_var, h_layer.weight.data.square().t())

                # relu
                normal_dist = Normal(0, 1)
                cdf_value = normal_dist.cdf(pred_mu / torch.sqrt(pred_var))
                pdf_value = torch.exp(normal_dist.log_prob(pred_mu / torch.sqrt(pred_var)))
                pred_mu_new = pred_mu * cdf_value + torch.sqrt(pred_var) * pdf_value
                pred_var_new = (pred_mu.square() + pred_var) * cdf_value + pred_mu * torch.sqrt(pred_var) * pdf_value - pred_mu_new.square()
                pred_mu, pred_var = pred_mu_new, pred_var_new

                # dropout
                pred_mu = pred_mu * (1 - p_drop)
                pred_var = pred_var * p_drop * (1 - p_drop) + pred_var * (1 - p_drop) ** 2 + pred_mu.square() * p_drop * (1 - p_drop)

            pred_mu = self.out_layer(pred_mu)
            pred_sigma = torch.sqrt(torch.mm(pred_var, self.out_layer.weight.data.square().t()))
        return pred_mu, pred_sigma

    def sample(self, x):
        pred_mu, pred_sigma = self.predict(x)
        z = torch.randn_like(pred_mu)
        return pred_sigma * z + pred_mu


class MCDropout(Module):
    def __init__(self,
                 in_dim: int = None,
                 out_dim: int = None,
                 hidden_dim: List[int] = None,
                 dropout_r: float = 0.3,
                 mc_samples: int = 1000):
        super(MCDropout, self).__init__()

        self.dropout_r = dropout_r
        self.mc_samples = mc_samples

        self.hidden_layers = nn.ModuleList()
        self.hidden_bn = nn.ModuleList()

        h_in = in_dim
        for h_out in hidden_dim:
            self.hidden_layers.append(layer.Linear(h_in, h_out))
            self.hidden_bn.append(nn.BatchNorm1d(h_out))
            h_in = h_out

        self.out_layer = layer.Linear(h_out, out_dim)

    def infer(self, x, sample=True):
        """
        x = torch.randn(200, 512, 30).cuda()
        """
        mask = self.training or sample
        n_samples, batch_size, _ = x.shape
        for h_layer, bn in zip(self.hidden_layers, self.hidden_bn):
            x = h_layer(x)
            x = bn(x.view(-1, bn.num_features)).view(n_samples, batch_size, bn.num_features)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout_r, training=mask)

        x = self.out_layer(x)
        # print([p.grad for p in list(self.out_layer.parameters()) if p.grad is not None])
        return x

    def forward(self, x):
        """
        x = torch.cat([features['idx'], features['macro']], dim=-1)
        x = torch.randn(512, 30).cuda()
        """
        x = x.unsqueeze(0).repeat(self.mc_samples, 1, 1)
        pred = self.infer(x, sample=True)

        pred_mu = torch.mean(pred, dim=0)
        with torch.set_grad_enabled(False):
            pred_sigma = torch.std(pred + 1e-6, dim=0, unbiased=False)

        return pred, pred_mu, pred_sigma


class MLPWithBN(Module):
    def __init__(self,
                 in_dim: int = None,
                 out_dim: int = None,
                 hidden_dim: List[int] = None):
        """
        :param in_dim: pred_mu + pred_sigma + prev_wgt + input =
        :param out_dim: wgt_mu + wgt_sigma
        """
        super(MLPWithBN, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_bn = nn.ModuleList()

        h_in = in_dim
        for h_out in hidden_dim:
            self.hidden_layers.append(layer.Linear(h_in, h_out))
            self.hidden_bn.append(nn.BatchNorm1d(h_out))
            h_in = h_out

        self.out_layer = layer.Linear(h_out, out_dim)

    def forward(self, x):

        for i in range(len(self.hidden_layers)):
            h_layer, bn = self.hidden_layers[i], self.hidden_bn[i]
            x = h_layer(x)
            x = bn(x)
            x = F.relu(x)

        x = self.out_layer(x)
        wgt_mu, wgt_logsigma = torch.chunk(x, 2, dim=-1)
        wgt_sigma = 1. * F.softplus(wgt_logsigma) + 1e-6

        return wgt_mu, wgt_sigma


class Attention(Module):
    def __init__(self,
                 d_k: int = None,
                 d_v: int = None,
                 d_model: int = None,
                 d_ff: int = None,
                 n_heads: int = None,
                 dropout: float = 0.3):
        super(Attention, self).__init__()
        self.encoder = layer.EncoderLayer(d_k, d_v, d_model, d_ff, n_heads, dropout=dropout)
        self.decoder = layer.DecoderLayer(d_k, d_v, d_model, d_ff, n_heads, dropout=dropout)

    def forward(self, x):
        """
        x shape: [n_batch, n_timesteps, d_model]
        """

        """
        x: [num_batch, num_timesteps_in, d_model]
        enc_context: -> [num_batch, num_timesteps_in, d_model]
        enc_self_attn: -> [num_batch, num_heads, num_timesteps_in, num_timesteps_in]
        """
        enc_context, enc_self_attn = self.encoder(x, True)

        """
        x[:, -1:, :] : [num_batch, num_timesteps_out, d_model]
        dec_context: -> [num_batch, num_timesteps_out, d_model]
        dec_self_attn: [num_batch, num_heads, num_timesteps_out, num_timesteps_out]
        dec_cross_attn: [num_batch, num_heads, num_timesteps_out, num_timesteps_in]
        """
        dec_context, dec_self_attn, dec_cross_attn = self.decoder(x[:, -1:, :], enc_context)

        return dec_context.view([len(x), -1])


class MyModel(pl.LightningModule):
    """
        dm = TestDataManager()
        train_sample = dm.sample('train')
        test_sample = dm.sample('test')
    """
    def __init__(self, model_cfg, exp_cfg, dm):
        super(MyModel, self).__init__()

        self.model_cfg = model_cfg
        self.exp_cfg = exp_cfg
        self.dm = dm

        # attentive models
        self.conv_emb = layer.ConvEmbeddingLayer(len(dm.features_list), model_cfg.d_model)
        self.attentive_model = Attention(**model_cfg.attention)

        # mc dropout models
        self.expected_return_estimator = MCDropout(**model_cfg.mcdropout)

        # allocator (non-cash only)
        self.strategies_allocator = MLPWithBN(**model_cfg.allocator)

        self.optim_state_dict = self.state_dict()
        self.current_stage = None

    def set_stage(self, stage=1):
        assert stage in [1, 2]
        if stage == 1:
            self.current_stage = 1
            self.lr = self.exp_cfg.pre_lr
            self.loss_wgt = self.model_cfg.pre_loss_wgt
        else:
            self.current_stage = 1
            self.lr = self.exp_cfg.lr
            self.loss_wgt = self.model_cfg.loss_wgt

    def forward(self, x):
        ########
        # defined parameter in configs
        base_weight = self.exp_cfg.base_weight
        use_guide_wgt_as_prev_x = self.exp_cfg.use_guide_wgt_as_prev_x
        ########
        if self.current_stage is None:
            print('do set_stage(stage={1|2}) before run')
            return None

        features, features_prev = x
        wgt_guide = torch.tensor(base_weight).repeat([len(features), 1]).type_as(features)
        with torch.set_grad_enabled(False):
            if features_prev is not None:
                wgt_prev, _, _, _ = self.infer(features_prev, wgt_guide)
                wgt_prev = self.noncash_to_all(wgt_prev, wgt_guide)
            else:
                wgt_prev = wgt_guide

        # x, pred_mu, pred_sigma = self.run(features, prev_x, n_samples)
        if use_guide_wgt_as_prev_x is True:
            wgt_mu, wgt_sigma, pred_mu, pred_sigma = self.infer(features, wgt_guide)
        else:
            wgt_mu, wgt_sigma, pred_mu, pred_sigma = self.infer(features, wgt_prev)

        dist = torch.distributions.Normal(loc=wgt_mu, scale=wgt_sigma)
        if self.training:
            wgt_ = wgt_mu + torch.randn_like(wgt_sigma) * wgt_sigma  # reparameterization trick
        else:
            wgt_ = wgt_mu

        wgt_ = self.noncash_to_all(wgt_, wgt_guide)

        return dist, wgt_, wgt_prev, pred_mu, pred_sigma, wgt_guide

    @profile
    def shared_step(self, batch):
        ########
        # defined parameter in configs
        loss_list = self.model_cfg.loss_list
        mdd_cp = self.exp_cfg.mdd_cp
        cost_rate = self.exp_cfg.cost_rate
        loss_wgt = self.loss_wgt
        ########

        features, labels, features_prev, labels_prev = self.data_dict_to_list(batch)
        dist, x, prev_x, pred_mu, pred_sigma, wgt_guide = self([features, features_prev])

        next_logy = labels['logy'].detach().clone()
        next_y = torch.exp(next_logy) - 1.

        losses_dict = dict()
        for i_loss, key in enumerate(loss_list):
            if key == 'y_pf':
                losses_dict[key] = -((x - wgt_guide) * next_y).sum()
            elif key == 'mdd_pf':
                losses_dict[key] = 10 * F.elu(-(x * next_y).sum(dim=1) - mdd_cp, 1e-6).sum()
            elif key == 'logy':
                losses_dict[key] = nn.MSELoss(reduction='sum')(pred_mu, labels['logy'])
            elif key == 'wgt_guide':
                losses_dict[key] = nn.KLDivLoss(reduction='sum')(torch.log(x), wgt_guide).sum()
            elif key == 'cost':
                if labels_prev is not None:
                    next_y_prev = torch.exp(labels_prev['logy']) - 1.
                    wgt_prev = prev_x * (1 + next_y_prev)
                    wgt_prev = wgt_prev / wgt_prev.sum(dim=1, keepdim=True)
                    losses_dict[key] = (torch.abs(x - wgt_prev) * cost_rate).sum().exp()
                else:
                    losses_dict[key] = (torch.abs(x - wgt_guide) * cost_rate).sum().exp()

            losses_dict[key] = losses_dict[key] * loss_wgt[key]

        losses = torch.tensor(0, dtype=torch.float32).to(labels['logy'].device)
        for key in losses_dict.keys():
            losses += losses_dict[key]

        additional = {'pred': x, 'next_y': next_y, 'guide': wgt_guide}
        return losses, losses_dict, additional

    def training_step(self, batch, batch_idx):
        losses, losses_dict, _ = self.shared_step(batch)
        self.log_dict({"loss": losses, "losses_dict": losses_dict})
        return {"loss": losses}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        losses, losses_dict, additional = self.shared_step(batch)
        return {"loss": losses, "losses_dict": losses_dict, "additional": additional}

    def train_dataloader(self) -> DataLoader:
        return self.dm.get_data_loader(self.exp_cfg.ii, 'train')

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        val_dataloader = self.dm.get_data_loader(self.exp_cfg.ii, 'eval')
        test_insample_dataloader = self.dm.get_data_loader(self.exp_cfg.ii, 'test_insample')
        test_dataloader = self.dm.get_data_loader(self.exp_cfg.ii, 'test')
        return [val_dataloader, test_insample_dataloader, test_dataloader]

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.dm.get_data_loader(self.exp_cfg.ii, 'test')

    def on_train_start(self) -> None:
        print('stage {} train start.'.format(self.current_stage))
        self.trainer.accelerator_backend.setup_optimizers(self)

    def configure_optimizers(self):
        return RAdam(self.parameters(), lr=self.lr)

    def validation_epoch_end(self, multiloader_outputs: List[Any]) -> None:
        val_losses = 0
        test_losses = 0

        for dataloader_i, outputs in enumerate(multiloader_outputs):
            if dataloader_i == 0:
                for i, batch in enumerate(outputs):
                    val_losses += batch['loss']
                    print("[valid_step_end]: (val) {} {} {}".format(i, batch['additional']['pred'].shape, batch['additional']['next_y'].shape))

            elif dataloader_i == 1:
                self.plot(outputs, 'test_insample')
                # for i, batch in enumerate(outputs):
                #     print("[valid_step_end]: (test_insample) {} {} {}".format(i, batch['additional']['wgt'].shape, batch['additional']['next_y'].shape))

            elif dataloader_i == 2:
                self.plot(outputs, 'test')
                # for i, batch in enumerate(outputs):
                #     test_losses += batch['loss']
                #     print("[valid_step_end]: (test) {} {} {}".format(i, batch['additional']['wgt'].shape, batch['additional']['next_y'].shape))

        # self.log_dict({"val_loss": val_losses, 'test_loss': test_losses})
        self.log_dict({"val_loss": val_losses})
        return {'val_loss': val_losses}

    def infer(self, x, wgt):
        """
        wgt = features['wgt']
        """
        # positional & conv embeding
        x_emb = self.conv_emb(x)

        # attentive models
        x_attn = self.attentive_model(x_emb)

        # expected return estimation
        _, pred_mu, pred_sigma = self.expected_return_estimator(x_emb[:, -1, :])

        # allocation
        x = torch.cat([pred_mu, pred_sigma, wgt, x_emb[:, -1, :], x_attn], dim=-1)
        wgt_mu, wgt_sigma = self.strategies_allocator(x)

        return wgt_mu, wgt_sigma, pred_mu, pred_sigma

    def noncash_to_all(self, wgt_, wgt_guide):
        ########
        # defined parameter in configs
        cash_idx = self.exp_cfg.cash_idx
        wgt_range_min = self.exp_cfg.wgt_range_min
        wgt_range_max = self.exp_cfg.wgt_range_max
        ########

        noncash_idx = np.delete(np.arange(wgt_guide.shape[1]), cash_idx)

        wgt_max_noncash = torch.tensor(wgt_range_max).to(wgt_)[noncash_idx]
        wgt_min_noncash = torch.tensor(wgt_range_min).to(wgt_)[noncash_idx]

        wgt_min_cash = torch.tensor(wgt_range_min).to(wgt_)[cash_idx]

        # 모델 산출 값만큼 반영하고,
        # wgt_ = torch.tanh(wgt_)
        # wgt_ = (1. + wgt_) * wgt_guide[:, noncash_idx]
        wgt_ = (1. + torch.tanh(wgt_)) * wgt_guide[:, noncash_idx]
        # wgt_ = torch.softmax(wgt_, dim=-1)

        # 자산별 min/max 맞춘 후
        # wgt_ = wgt_min_noncash + wgt_ * (wgt_max_noncash - wgt_min_noncash)
        wgt_ = torch.min(torch.max(wgt_, wgt_min_noncash), wgt_max_noncash)

        # 최소 cash 비율 맞추고
        # wgt_rf = torch.max(1 - wgt_.sum(-1, keepdim=True), wgt_min_cash)
        # wgt_ = wgt_ / (wgt_.sum(-1, keepdim=True) + 1e-6) * (1 - wgt_rf)
        wgt_rf = torch.max(1 - wgt_.sum(-1, keepdim=True),
                           torch.zeros(wgt_.shape[0], 1, device=wgt_.device))

        # 이어붙인다
        wgt_ = torch.cat([wgt_[:, :cash_idx], wgt_rf, wgt_[:, cash_idx:]], dim=-1) + 1e-3

        # 이미 합은 1이어야 하지만 혹시 몰라 조정
        wgt_ = wgt_ / (wgt_.sum(-1, keepdim=True) + 1e-6)
        return wgt_

    def data_dict_to_list(self, data_dict):
        features = data_dict.get('features')
        labels = data_dict.get('labels')

        features_prev = data_dict.get('features_prev')
        labels_prev = data_dict.get('labels_prev')

        return features, labels, features_prev, labels_prev

    def adversarial_noise(self, data_dict, losses_wgt_fixed=None):
        features, labels, features_prev, labels_prev = self.data_dict_to_list(data_dict)

        features.requires_grad = True
        for key in labels.keys():
            labels[key].requires_grad = True

        forward_result = self.forward_with_loss(data_dict, mc_samples=100, losses_wgt_fixed=losses_wgt_fixed)

        if forward_result is False:
            return False

        pred, losses, _, _, _ = forward_result

        losses.backward(retain_graph=True)

        features_grad = torch.sign(features.grad)
        sample_sigma = torch.std(features + 1e-6, axis=[0], keepdims=True)
        eps = 0.05
        scaled_eps = eps * sample_sigma  # shape: [1, 1, n_features]

        features_perturbed = features + scaled_eps * features_grad
        features.grad.zero_()

        return features_perturbed

    def plot(self, outputs: List[dict], mode):
        ########
        # defined parameter in configs
        ii = self.exp_cfg.ii
        cost_rate = self.exp_cfg.cost_rate
        ########

        plot_data = dict()
        for key in ['next_y', 'pred', 'guide']:
            plot_data[key] = tu.np_ify(torch.cat([batch['additional'][key] for batch in outputs], dim=0))

        self.dm.plot(plot_data, ii, cost_rate)



        y_next = data_for_plot['next_y'][selected_sampling, :]
        wgt_result_calc = wgt_result[selected_sampling, :]
        wgt_guide_calc = wgt_guide[selected_sampling, :]


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

        fig.savefig(os.path.join(outpath_plot, '{}_test_wgt_{}.png'.format(ep, suffix)))
        plt.close(fig)

    def save_to_optim(self):
        self.optim_state_dict = copy.deepcopy(self.state_dict())

    def load_from_optim(self):
        self.load_state_dict(self.optim_state_dict)



