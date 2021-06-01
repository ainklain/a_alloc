
# renewed from model_attn
from omegaconf import DictConfig
from typing import List, Union, Any
import torch
from torch import nn
from torch.nn import  Module, functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.distributions import Normal
import pytorch_lightning as pl

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import layer
import torch_utils as tu
from optimizers.optimizer_v2 import RAdam
from dataprocess.dataset_v2 import DatasetManager


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


class MCDModel(pl.LightningModule):
    """
        dm = TestDataManager()
        train_sample = dm.sample('train')
        test_sample = dm.sample('test')
    """
    def __init__(self,
                 model_cfg: DictConfig,
                 exp_cfg: DictConfig,
                 dm: DatasetManager):
        super(MCDModel, self).__init__()

        self.model_cfg = model_cfg
        self.exp_cfg = exp_cfg
        self.dm = dm

        # positional & conv embeding
        self.conv_emb = layer.ConvEmbeddingLayer(len(dm.features_list), model_cfg.d_model)

        self.model = MCDropout(**model_cfg.mcdropout)

    def set_stage(self, stage, trainer_cfg):
        assert stage in [1, 2]
        if stage == 1:
            self.current_stage = stage
            self.lr = trainer_cfg.stage1.lr
            self.loss_wgt = trainer_cfg.stage1.loss_wgt
        else:
            self.current_stage = stage
            self.lr = trainer_cfg.stage2.lr
            self.loss_wgt = trainer_cfg.stage2.loss_wgt

    def forward(self, x):
        # positional & conv embeding
        x_emb = self.conv_emb(x)

        # expected return estimation
        _, pred_mu, pred_sigma = self.model(x_emb[:, -1, :])

        return pred_mu, pred_sigma

    def shared_step(self, batch):
        features, labels, features_prev, labels_prev = self.data_dict_to_list(batch)
        pred_mu, pred_sigma = self(features)

        losses = nn.MSELoss(reduction='sum')(pred_mu, labels['logy'])

        additional = {'pred_mu': pred_mu, 'pred_sigma': pred_sigma, 'label': labels['logy']}
        return losses, additional

    def training_step(self, batch, batch_idx):
        losses, _ = self.shared_step(batch)
        return {"loss": losses}

    def validation_step(self, batch, batch_idx):
        losses, additional = self.shared_step(batch)
        return {"loss": losses, "additional": additional}

    def train_dataloader(self) -> DataLoader:
        return self.dm.get_data_loader(self.exp_cfg.ii, 'train')

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        test_insample_dataloader = self.dm.get_data_loader(self.exp_cfg.ii, 'test_insample')
        return test_insample_dataloader

    def configure_optimizers(self):
        optimizer = RAdam(self.parameters(), lr=self.lr)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=0, verbose=True)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        return {"optimizer": optimizer,
                "scheduler": scheduler,
                "interval": "epoch",
                # "monitor": "metric_to_track"
                }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        val_losses = 0
        for i, batch in enumerate(outputs):
            val_losses += batch['loss']
        if self.current_epoch % 100 == 0:
            self.plot_result(outputs, self.current_epoch)

        print("pred: \n", outputs[0]['additional']['pred_mu'],
              "label: \n", outputs[0]['additional']['label'])
        self.log_dict({"val_loss": val_losses,
                       'step': self.current_epoch})

        return {'val_loss': val_losses}

    def data_dict_to_list(self, data_dict):
        features = data_dict.get('features')
        labels = data_dict.get('labels')

        features_prev = data_dict.get('features_prev')
        labels_prev = data_dict.get('labels_prev')

        return features, labels, features_prev, labels_prev

    def custom_histogram_adder(self):
        # iterating through all parameters
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def plot_result(self, outputs: List[dict], current_epoch):
        ######################
        # calculate portfolio result through datamanager (dataframe)
        ######################
        plot_data = dict()
        for key in ['pred_mu', 'pred_sigma', 'label']:
            plot_data[key] = tu.np_ify(torch.cat([batch['additional'][key] for batch in outputs], dim=0))
            pd.DataFrame(plot_data[key]).to_csv(os.path.join(self.exp_cfg.outpath, 'df_{}_{}.csv'.format(current_epoch, key)))

        ######################
        # plot
        ######################

        outpath_plot = os.path.join(self.exp_cfg.outpath, 'plot')
        os.makedirs(outpath_plot, exist_ok=True)

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        # ax = plt.gca()

        # label
        df = pd.DataFrame({'true': plot_data['label'][:, 0],
                           'pred_mu': plot_data['pred_mu'][:, 0],
                           'lower': plot_data['pred_mu'][:, 0] - plot_data['pred_sigma'][:, 0],
                           'upper': plot_data['pred_mu'][:, 0] + plot_data['pred_sigma'][:, 0]})
        df.plot(ax=ax1)

        fig.savefig(os.path.join(outpath_plot, 'sample_{}.png'.format(current_epoch)))
        plt.close(fig)


