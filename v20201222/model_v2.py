# renewed from model_attn

import copy
from collections import OrderedDict
import os
import numpy as np
import torch
from torch import nn, distributions
from torch.nn import init, Module, functional as F
import layer
from torch.distributions import Categorical


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


class TestDataManager:
    """
    dm = TestDataManager()
    x = dm.sample('train')
    """
    def __init__(self):
        from main_v2 import Configs
        from dataset_v2 import DatasetManager, AplusData, MacroData
        c = Configs('testdatamanager')
        data_list = [AplusData(), MacroData()]
        dataset_type = 'multitask'
        test_days = c.test_days
        batch_size = c.batch_size
        self.dm = DatasetManager(data_list, test_days, batch_size)

    def get_loader(self, base_i=1000, mode='train'):
        return self.dm.get_data_loader(base_i, mode)

    def sample(self, mode='train'):
        dl = self.get_loader(1000, mode)
        return next(iter(dl))


def test_model():
    dm = TestDataManager()
    sample_train = dm.sample('train')
    features_prev = sample_train['features_prev']
    features = sample_train['features']
    labels = sample_train['labels']
    labels_prev = sample_train['labels_prev']

    ##
    from main_v2 import Configs
    c = Configs('test')
    model = MyModel(len(dm.dm.features_list), len(dm.dm.labels_list), configs=c)
    out = model.predict(features, features_prev, True, 100)
    model.forward_with_loss(features, labels, features_prev=features_prev, labels_prev=labels_prev)
    ##

    base_weight = [0.69, 0.3, 0.1, 0.01]
    n_features = x.shape[-1]
    n_asset = 4

    n_heads = 4
    d_model = 64
    d_ff = 64
    d_k = d_v = d_model // n_heads

    # conv embedding
    from layer import ConvEmbeddingLayer
    conv_emb = ConvEmbeddingLayer(x.shape[-1], d_model)
    x_emb = conv_emb(x)

    # encoder with self attention
    from layer import MultiHeadAttention, PoswiseFeedForwardNet
    self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout=0.5)
    enc_pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout=0.1)
    x_context, attn = self_attn(x_emb, x_emb, x_emb, False)
    x_context = enc_pos_ffn(x_context)

    from layer import EncoderLayer
    encoder = EncoderLayer(d_k, d_v, d_model, d_ff, n_heads, dropout=0.3)
    x_context2, attn2 = encoder(x_emb, True)

    # decoder with attention
    y_emb = x_emb[:, -1:, :]
    dec_self_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout=0.5)
    dec_cross_attn = MultiHeadAttention(d_k, d_v, d_model, n_heads, dropout=0.5)
    pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout=0.1)
    y_context, attn = dec_self_attn(y_emb, y_emb, y_emb, True)
    y_context, attn = dec_cross_attn(y_context, x_context, x_context, False)
    y_context = pos_ffn(y_context).squeeze()


    from layer import DecoderLayer
    decoder = DecoderLayer(d_k, d_v, d_model, d_ff, n_heads, dropout=0.3)
    y_context2, dec_self_attn, dec_enc_attn = decoder(y_emb, x_context2)
    y_context2 = y_context2.squeeze()


    guide_wgt = torch.FloatTensor([base_weight]).repeat(len(x), 1).to(x.device)

    prev_x = guide_wgt

    x_er = x[:, -1, :]
    expected_return_estimator = ExpectedReturnEstimator(in_dim=n_features, out_dim=n_asset, hidden_dim=[64, 64])
    pred, pred_mu, pred_sigma = expected_return_estimator.run(x_er, 100)

    strategies_allocator = StrategiesAllocator(d_model + n_features + n_asset * 3, n_asset * 2, hidden_dim=[64, 64])

    x_total = torch.cat([pred_mu, pred_sigma, y_context, prev_x, x_er], dim=-1)
    wgt_mu, wgt_sigma = strategies_allocator(x_total)

    wgt_mu = (1. + wgt_mu) * guide_wgt

    # cash 제한 풀기
    noncash_idx = np.delete(np.arange(wgt_mu.shape[1]), 3)
    wgt_mu_rf = torch.max(1 - wgt_mu[:, noncash_idx].sum(-1, keepdim=True),
                          torch.zeros(wgt_mu.shape[0], 1, device=wgt_mu.device))
    wgt_mu = torch.cat([wgt_mu[:, noncash_idx], wgt_mu_rf], dim=-1)

    wgt_mu = wgt_mu / (wgt_mu.sum(-1, keepdim=True) + 1e-6)
    dist = torch.distributions.Normal(loc=wgt_mu, scale=wgt_sigma)
    is_train = True
    if is_train:
        # pred = dist.rsample().clamp(0.01, 0.99)
        pred = (wgt_mu + torch.randn_like(wgt_sigma) * wgt_sigma).clamp(0.01, 0.99) # reparameterization trick
        pred = pred / (pred.sum(-1, keepdim=True) + 1e-6)
    else:
        pred = wgt_mu


    with torch.set_grad_enabled(False):
        next_logy = torch.empty_like(y['logy']).to('cpu')
        next_logy.copy_(y['logy'])
        # next_logy = torch.exp(next_logy) - 1.
        if is_train:
            random_setting = torch.empty_like(next_logy).to('cpu').uniform_()
            flip_mask = random_setting < 0.1
            next_logy[flip_mask] = -next_logy[flip_mask]

            sampling_mask = (random_setting >= 0.1) & (
                    random_setting < 0.2)
            next_logy[sampling_mask] = y['mu'][sampling_mask] + \
                                       (torch.randn_like(next_logy) * y['sigma'])[sampling_mask]

        next_y = torch.exp(next_logy) - 1.
        # next_y = next_logy


        losses_dict = dict()
        losses_wgt_dict = dict()
        # losses_dict['y_pf'] = -(x * labels['logy']).sum()

        losses_vars = torch.exp(self.loss_logvars)
        for i_loss, key in enumerate(['y_pf', 'mdd_pf', 'logy', 'wgt_guide']):
            if key == 'y_pf':
                losses_dict[key] = -((pred - guide_wgt) * next_y).sum()
            elif key == 'mdd_pf':
                losses_dict[key] = 10 * F.elu(-(pred * next_y).sum(dim=1) - 0.05, 1e-6).sum()
            elif key == 'logy':
                losses_dict[key] = nn.MSELoss(reduction='sum')(pred_mu, y['logy'])
            elif key == 'wgt_guide':
                losses_dict[key] = nn.KLDivLoss(reduction='sum')(torch.log(pred), guide_wgt).sum()
            elif key == 'cost':
                if labels_prev is not None:
                    next_y_prev = torch.exp(y['logy'][:, -1, :]) - 1.
                    wgt_prev = prev_x * (1 + next_y_prev)
                    wgt_prev = wgt_prev / wgt_prev.sum(dim=1, keepdim=True)
                    losses_dict[key] = (torch.abs(x - wgt_prev) * 0.003).sum()
                else:
                    losses_dict[key] = (torch.abs(x - guide_wgt) * 0.003).sum()
            elif key == 'entropy':
                if c.max_entropy:
                    losses_dict[key] = -dist.entropy().sum(dim=1).mean()
                else:
                    losses_dict[key] = dist.entropy().sum(dim=1).mean()
            # losses_dict['entropy'] = HLoss()(x)
            losses_dict[key] = losses_dict[key] / (losses_vars[i_loss] + 1e-6) + 0.5 * self.loss_logvars[i_loss]


class ExpectedReturnEstimator(Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(ExpectedReturnEstimator, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.hidden_bn = nn.ModuleList()

        h_in = in_dim
        for h_out in hidden_dim:
            self.hidden_layers.append(layer.Linear(h_in, h_out))
            self.hidden_bn.append(nn.BatchNorm1d(h_out))
            h_in = h_out

        self.out_layer = layer.Linear(h_out, out_dim)

    def forward(self, x, sample=True, dropout=0.3):
        """
        x = torch.randn(200, 512, 30).cuda()
        """
        mask = self.training or sample
        n_samples, batch_size, _ = x.shape
        for h_layer, bn in zip(self.hidden_layers, self.hidden_bn):
            x = h_layer(x)
            x = bn(x.view(-1, bn.num_features)).view(n_samples, batch_size, bn.num_features)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=dropout, training=mask)

        x = self.out_layer(x)
        # print([p.grad for p in list(self.out_layer.parameters()) if p.grad is not None])
        return x

    def run(self, x, mc_samples, dropout=0.3):
        """
        x = torch.cat([features['idx'], features['macro']], dim=-1)
        x = torch.randn(512, 30).cuda()
        """
        x = x.unsqueeze(0).repeat(mc_samples, 1, 1)
        pred = self.forward(x, sample=True, dropout=dropout)

        pred_mu = torch.mean(pred, dim=0)
        with torch.set_grad_enabled(False):
            pred_sigma = torch.std(pred + 1e-6, dim=0, unbiased=False)

        return pred, pred_mu, pred_sigma


class StrategiesAllocator(Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        """
        :param in_dim: pred_mu + pred_sigma + prev_wgt + input =
        :param out_dim: wgt_mu + wgt_sigma
        """
        super(StrategiesAllocator, self).__init__()

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
        # wgt_mu = 0.99 * torch.tanh(wgt_mu) + 1e-6
        # wgt_sigma = 0.2 * F.softplus(wgt_logsigma) + 1e-6

        # wgt_mu = 2. * torch.tanh(wgt_mu) + 1e-6
        wgt_sigma = 1. * F.softplus(wgt_logsigma) + 1e-6

        return wgt_mu, wgt_sigma


class AttentiveLatentModel(Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, dropout=0.3):
        super(AttentiveLatentModel, self).__init__()
        self.encoder = layer.EncoderLayer(d_k, d_v, d_model, d_ff, n_heads, dropout=dropout)
        self.decoder = layer.DecoderLayer(d_k, d_v, d_model, d_ff, n_heads, dropout=dropout)

    def forward(self, x):
        """
        x shape: [n_batch, n_timesteps, d_model]
        """
        enc_context, enc_self_attn = self.encoder(x, True)
        dec_context, dec_self_attn, dec_cross_attn = self.decoder(x[:, -1:, :], enc_context)

        return dec_context.view([len(x), -1])


class MyModel(Module):
    """
        dm = TestDataManager()
        train_sample = dm.sample('train')
        test_sample = dm.sample('test')
    """
    def __init__(self, n_features, n_assets, configs, ):
        super(MyModel, self).__init__()
        c = configs
        self.in_dim = n_features
        self.out_dim = n_assets
        self.c = c

        # attentive model
        self.conv_emb = layer.ConvEmbeddingLayer(n_features, c.d_model)
        self.attentive_model = AttentiveLatentModel(c.d_k, c.d_v, c.d_model, c.d_ff, c.n_heads, c.dropout_r)

        # mc dropout model
        # self.expected_return_estimator = ExpectedReturnEstimator(n_features, n_assets, c.hidden_dim)
        self.expected_return_estimator = ExpectedReturnEstimator(c.d_model, n_assets, c.hidden_dim)

        # allocator (non-cash only)
        self.strategies_allocator = StrategiesAllocator(in_dim=n_assets * 3 + c.d_model * 2, out_dim=(n_assets-1) * 2, hidden_dim=c.alloc_hidden_dim)

        self.loss_logvars = torch.nn.Parameter(torch.zeros(len(c.loss_list), dtype=torch.float32, requires_grad=True))
        self.optim_state_dict = self.state_dict()

    def random_guide_weight(self, base_weight, batch_size, is_train, random_r=0.):
        with torch.set_grad_enabled(False):
            self.guide_weight = torch.FloatTensor([base_weight])
            if is_train and np.random.rand() <= random_r:
                guide_wgt = torch.rand(batch_size, self.out_dim)
                guide_wgt = guide_wgt / guide_wgt.sum(dim=1, keepdim=True)
            else:
                guide_wgt = self.guide_weight.repeat(batch_size, 1)

        return guide_wgt

    def forward(self, x, wgt, mc_samples):
        """
        wgt = features['wgt']
        """
        # positional & conv embeding
        x_emb = self.conv_emb(x)

        # attentive model
        x_attn = self.attentive_model(x_emb)

        # expected return estimation
        # _, pred_mu, pred_sigma = self.expected_return_estimator.run(x[:, -1, :], mc_samples=mc_samples)
        _, pred_mu, pred_sigma = self.expected_return_estimator.run(x_emb[:, -1, :], mc_samples=mc_samples)
        # print(pred_mu.exp().mean(dim=0), pred_sigma.mean(dim=0))
        # allocation
        x = torch.cat([pred_mu, pred_sigma, wgt, x_emb[:, -1, :], x_attn], dim=-1)
        wgt_mu, wgt_sigma = self.strategies_allocator(x)

        return wgt_mu, wgt_sigma, pred_mu, pred_sigma

    def noncash_to_all(self, wgt_, wgt_guide):
        c = self.c
        noncash_idx = np.delete(np.arange(wgt_guide.shape[1]), c.cash_idx)

        wgt_max_noncash = torch.tensor(c.wgt_range_max).to(wgt_)[noncash_idx]
        wgt_min_noncash = torch.tensor(c.wgt_range_min).to(wgt_)[noncash_idx]

        wgt_min_cash = torch.tensor(c.wgt_range_min).to(wgt_)[c.cash_idx]

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
        wgt_ = torch.cat([wgt_[:, :c.cash_idx], wgt_rf, wgt_[:, c.cash_idx:]], dim=-1) + 1e-3

        # 이미 합은 1이어야 하지만 혹시 몰라 조정
        wgt_ = wgt_ / (wgt_.sum(-1, keepdim=True) + 1e-6)
        return wgt_

    def predict(self, features, features_prev, is_train, mc_samples):
        c = self.c
        wgt_guide = self.random_guide_weight(c.base_weight, len(features), is_train, c.random_guide_weight).to(features.device)

        with torch.set_grad_enabled(False):
            if features_prev is not None:
                # prev_x, _, _ = self.run(features_prev, features['wgt'], n_samples)
                wgt_prev, _, _, _ = self.forward(features_prev, wgt_guide, mc_samples)
                wgt_prev = self.noncash_to_all(wgt_prev, wgt_guide)
            else:
                wgt_prev = wgt_guide

        # x, pred_mu, pred_sigma = self.run(features, prev_x, n_samples)
        if c.use_guide_wgt_as_prev_x is True:
            wgt_mu, wgt_sigma, pred_mu, pred_sigma = self.forward(features, wgt_guide, mc_samples)
        else:
            wgt_mu, wgt_sigma, pred_mu, pred_sigma = self.forward(features, wgt_prev, mc_samples)

        dist = torch.distributions.Normal(loc=wgt_mu, scale=wgt_sigma)
        if is_train:
            # wgt_ = dist.rsample()
            wgt_ = wgt_mu + torch.randn_like(wgt_sigma) * wgt_sigma  # reparameterization trick
        else:
            wgt_ = wgt_mu

        wgt_ = self.noncash_to_all(wgt_, wgt_guide)
        return dist, wgt_, wgt_prev, pred_mu, pred_sigma, wgt_guide

    def data_dict_to_list(self, data_dict):
        features = data_dict.get('features')
        labels = data_dict.get('labels')

        features_prev = data_dict.get('features_prev')
        labels_prev = data_dict.get('labels_prev')

        return features, labels, features_prev, labels_prev

    @profile
    def forward_with_loss(self, data_dict,
                          mc_samples=None,
                          is_train=True,
                          losses_wgt_fixed=None,
                          ):
        c = self.c

        if mc_samples is None:
            mc_samples = c.mc_samples

        features, labels, features_prev, labels_prev = self.data_dict_to_list(data_dict)
        features, labels, features_prev, labels_prev = tu.to_device(tu.device, [features, labels, features_prev, labels_prev ])
        dist, x, prev_x, pred_mu, pred_sigma, guide_wgt = self.predict(features, features_prev, is_train, mc_samples)

        # debugging code (nan value)
        if torch.isnan(x).sum() > 0:
        #     for n, p in list(self.named_parameters()):
        #         print(n, '\nval:\n', p, '\ngrad:\n', p.grad)

            return False

        if labels is not None:
            with torch.set_grad_enabled(False):
                next_logy = torch.empty_like(labels['logy']).to(labels['logy'].device)
                next_logy.copy_(labels['logy'])
                # next_logy = torch.exp(next_logy) - 1.
                if is_train:
                    random_setting = torch.empty_like(next_logy).to(tu.device).uniform_()
                    flip_mask = random_setting < c.random_flip
                    next_logy[flip_mask] = -next_logy[flip_mask]

                    sampling_mask = (random_setting >= c.random_flip) & (
                                random_setting < (c.random_flip + c.random_label))
                    next_logy[sampling_mask] = labels['mu'][sampling_mask] + \
                                               (torch.randn_like(next_logy) * labels['sigma'])[sampling_mask]

                # next_y = next_logy
                next_y = torch.exp(next_logy) - 1.

            losses_dict = dict()
            # losses_dict['y_pf'] = -(x * labels['logy']).sum()

            losses_vars = torch.exp(self.loss_logvars)
            for i_loss, key in enumerate(c.loss_list):
                if key == 'y_pf':
                    losses_dict[key] = -((x - guide_wgt) * next_y).sum()
                elif key == 'mdd_pf':
                    losses_dict[key] = 10 * F.elu(-(x * next_y).sum(dim=1) - c.mdd_cp, 1e-6).sum()
                elif key == 'logy':
                    losses_dict[key] = nn.MSELoss(reduction='sum')(pred_mu, labels['logy'])
                elif key == 'wgt_guide':
                    losses_dict[key] = nn.KLDivLoss(reduction='sum')(torch.log(x), guide_wgt).sum()
                elif key == 'cost':
                    if labels_prev is not None:
                        next_y_prev = torch.exp(labels_prev['logy']) - 1.
                        wgt_prev = prev_x * (1 + next_y_prev)
                        wgt_prev = wgt_prev / wgt_prev.sum(dim=1, keepdim=True)
                        losses_dict[key] = (torch.abs(x - wgt_prev) * c.cost_rate).sum().exp()
                    else:
                        losses_dict[key] = (torch.abs(x - guide_wgt) * c.cost_rate).sum().exp()
                elif key == 'entropy':
                    if c.max_entropy:
                        losses_dict[key] = -dist.entropy().sum(dim=1).mean()
                    else:
                        losses_dict[key] = dist.entropy().sum(dim=1).mean()
                # losses_dict['entropy'] = HLoss()(x)
                if losses_wgt_fixed is not None:
                    losses_dict[key] = losses_dict[key] * losses_wgt_fixed[key]
                else:
                    losses_dict[key] = losses_dict[key] / (losses_vars[i_loss] + 1e-6) + 0.5 * self.loss_logvars[i_loss]

            losses = torch.tensor(0, dtype=torch.float32).to(labels['logy'].device)
            for key in losses_dict.keys():
                losses += losses_dict[key]
        else:
            losses = None
            losses_dict = None

        return x, losses, pred_mu, pred_sigma, losses_dict

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

    def save_to_optim(self):
        self.optim_state_dict = copy.deepcopy(self.state_dict())

    def load_from_optim(self):
        self.load_state_dict(self.optim_state_dict)


class MyModel_original(Module):
    def __init__(self, in_dim, out_dim, configs, ):
        super(MyModel_original, self).__init__()
        c = configs
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.c = c

        if c.base_weight is not None:
            self.guide_weight = torch.FloatTensor([c.base_weight])
        else:
            self.guide_weight = torch.ones(1, out_dim, dtype=torch.float32) / out_dim
        self.hidden_layers = nn.ModuleList()
        self.hidden_bn = nn.ModuleList()

        h_in = in_dim
        for h_out in c.hidden_dim:
            self.hidden_layers.append(layer.Linear(h_in, h_out))
            self.hidden_bn.append(nn.BatchNorm1d(h_out))
            h_in = h_out

        self.out_layer = layer.Linear(h_out, out_dim)

        # asset allocation
        self.aa_hidden_layer = layer.Linear(out_dim * 3 + in_dim, (out_dim * 3 + in_dim) // 2)
        self.aa_hidden_layer2 = layer.Linear((out_dim * 3 + in_dim) // 2, out_dim * 2)
        self.aa_bn = nn.BatchNorm1d((out_dim * 3 + in_dim) // 2)
        self.aa_bn2 = nn.BatchNorm1d(out_dim * 2)

        self.aa_out_layer = layer.Linear(out_dim * 2, out_dim * 2)

        self.loss_func_logy = nn.MSELoss()

        self.optim_state_dict = self.state_dict()
        # self.optim_

    # def init_weight(self):
    #     return torch.ones_like()

    def forward(self, x, sample=True):
        """
        x = torch.randn(200, 512, 30).cuda()
        """
        c = self.c
        mask = self.training or sample
        n_samples, batch_size, _ = x.shape
        for h_layer, bn in zip(self.hidden_layers, self.hidden_bn):
            x = h_layer(x)

            x = bn(x.view(-1, bn.num_features)).view(n_samples, batch_size, bn.num_features)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=c.dropout_r, training=mask)

        x = self.out_layer(x)
        # print([p.grad for p in list(self.out_layer.parameters()) if p.grad is not None])
        return x

    def sample_predict(self, x, mc_samples):
        """
        x = torch.randn(512, 30)
        """
        # self.train()
        # Just copies type from x, initializes new vector
        # predictions = x.data.new(n_samples, x.shape[0], self.out_dim)
        #
        # for i in range(n_samples):
        #     y = self.forward(x, sample=True)
        #     predictions[i] = y

        predictions = x.unsqueeze(0).repeat(mc_samples, 1, 1)
        predictions = self.forward(predictions, sample=True)
        return predictions

    def adversarial_noise(self, features, labels, features_prev=None, labels_prev=None):
        for key in features.keys():
            features[key].requires_grad = True

        for key in labels.keys():
            labels[key].requires_grad = True

        pred, losses, _, _, _ = self.forward_with_loss(features, labels, mc_samples=100, losses_wgt_fixed=None,
                                                       features_prev=features_prev, labels_prev=labels_prev)

        losses.backward(retain_graph=True)
        features_grad = dict()
        features_perturbed = dict()
        for key in features.keys():
            features_grad[key] = torch.sign(features[key].grad)
            sample_sigma = torch.std(features[key] + 1e-6, axis=[0], keepdims=True)
            eps = 0.05
            scaled_eps = eps * sample_sigma  # [1, 1, n_features]

            features_perturbed[key] = features[key] + scaled_eps * features_grad[key]
            features[key].grad.zero_()

        return features_perturbed

    def run(self, features, wgt, mc_samples):
        """
        wgt = features['wgt']
        """
        x = torch.cat([features['idx'], features['macro']], dim=-1)
        pred = self.sample_predict(x, mc_samples=mc_samples)
        pred_mu = torch.mean(pred, dim=0)
        with torch.set_grad_enabled(False):
            pred_sigma = torch.std(pred + 1e-6, dim=0, unbiased=False)

        x = torch.cat([pred_mu, pred_sigma, wgt, x], dim=-1)
        x = self.aa_hidden_layer(x)
        x = self.aa_bn(x)
        x = F.relu(x)
        x = self.aa_hidden_layer2(x)
        x = self.aa_bn2(x)
        x = F.relu(x)
        x = self.aa_out_layer(x)
        wgt_mu, wgt_logsigma = torch.chunk(x, 2, dim=-1)
        wgt_mu = 0.99 * torch.tanh(wgt_mu) + 1e-6
        wgt_sigma = 0.2 * F.softplus(wgt_logsigma) + 1e-6

        # x = F.sigmoid(x) + 0.001
        # x = x / x.sum(dim=1, keepdim=True)

        # return x, pred_mu, pred_sigma

        return wgt_mu, wgt_sigma, pred_mu, pred_sigma

    @profile
    def forward_with_loss(self, features, labels=None,
                          mc_samples=1000,
                          losses_wgt_fixed=None,
                          features_prev=None,
                          labels_prev=None,
                          is_train=True,
                          ):
        """
        t = 2000; mc_samples = 200; is_train = True
        dataset = {'train': None, 'eval': None, 'test': None, 'test_insample': None}
        dataset['train'], dataset['eval'], dataset['test'], (dataset['test_insample'], insample_boundary), guide_date = sampler.get_batch(t)

        dataset['train'], train_n_samples = dataset['train']

        dataset['train'] = tu.to_device(tu.device, to_torch(dataset['train']))
        train_features_prev, train_labels_prev, train_features, train_labels = dataset['train']
        features_prev, labels_prev, features, labels = dict(), dict(), dict(), dict()
        sampled_batch_idx = np.random.choice(np.arange(train_n_samples), c.batch_size)
        for key in train_features.keys():
            features_prev[key] = train_features_prev[key][sampled_batch_idx]
            features[key] = train_features[key][sampled_batch_idx]

        for key in train_labels.keys():
            labels_prev[key] = train_labels_prev[key][sampled_batch_idx]
            labels[key] = train_labels[key][sampled_batch_idx]

        """
        # features, labels=  train_features, train_labels

        # x = torch.cat([features['idx'], features['macro']], dim=-1)
        # pred = self.sample_predict(x, n_samples=n_samples)
        # pred_mu = torch.mean(pred, dim=0)
        # pred_sigma = torch.std(pred, dim=0)
        #
        #
        # x = torch.cat([pred_mu, pred_sigma, features['wgt']], dim=-1)
        # x = self.aa_hidden_layer(x)
        # x = F.relu(x)
        # x = self.aa_out_layer(x)
        # # x = F.elu(x + features['wgt'], 0.01) + 0.01 + min_wgt
        # # x = F.softmax(x)
        # x = F.sigmoid(x) + 0.001
        # x = x / x.sum(dim=1, keepdim=True)

        c = self.c

        with torch.set_grad_enabled(False):
            if is_train:
                if np.random.rand() > c.random_guide_weight:
                    guide_wgt = self.guide_weight.repeat(len(features['wgt']), 1).to(features['wgt'].device)
                else:
                    guide_wgt = torch.rand_like(features['wgt']).to(features['wgt'].device)
                    guide_wgt = guide_wgt / guide_wgt.sum(dim=1, keepdim=True)
            else:
                guide_wgt = self.guide_weight.repeat(len(features['wgt']), 1).to(features['wgt'].device)

        if features_prev is not None:
            with torch.set_grad_enabled(False):
                # prev_x, _, _ = self.run(features_prev, features['wgt'], n_samples)
                prev_x, _, _, _ = self.run(features_prev, guide_wgt, mc_samples)
                prev_x = (1. + prev_x) * guide_wgt
                prev_x = prev_x / (prev_x.sum(-1, keepdim=True) + 1e-6)
        else:
            prev_x = features['wgt']

        # x, pred_mu, pred_sigma = self.run(features, prev_x, n_samples)
        if c.use_guide_wgt_as_prev_x is True:
            wgt_mu, wgt_sigma, pred_mu, pred_sigma = self.run(features, guide_wgt, mc_samples)
        else:
            wgt_mu, wgt_sigma, pred_mu, pred_sigma = self.run(features, prev_x, mc_samples)

        wgt_mu = (1. + wgt_mu) * guide_wgt

        # cash 제한 풀기
        noncash_idx = np.delete(np.arange(wgt_mu.shape[1]), c.cash_idx)
        wgt_mu_rf = torch.max(1 - wgt_mu[:, noncash_idx].sum(-1, keepdim=True),
                              torch.zeros(wgt_mu.shape[0], 1, device=wgt_mu.device))
        wgt_mu = torch.cat([wgt_mu[:, noncash_idx], wgt_mu_rf], dim=-1)

        wgt_mu = wgt_mu / (wgt_mu.sum(-1, keepdim=True) + 1e-6)
        dist = torch.distributions.Normal(loc=wgt_mu, scale=wgt_sigma)
        if is_train:
            x = dist.rsample().clamp(0.01, 0.99)
            x = x / (x.sum(-1, keepdim=True) + 1e-6)
        else:
            x = wgt_mu

        if torch.isnan(x).sum() > 0:
            for n, p in list(self.named_parameters()):
                print(n, '\nval:\n', p, '\ngrad:\n', p.grad)

            return False

        # guide_wgt = self.guide_weight.repeat(len(x), 1)
        # if is_train and self.random_guide_weight > 0:
        #     replace_idx = np.random.choice(np.arange(len(x)), int(len(x) * self.random_guide_weight), replace=False)
        #     guide_wgt[replace_idx] = torch.rand(len(replace_idx), guide_wgt.shape[1])
        # guide_wgt = guide_wgt.to(x.device)

        if labels is not None:

            with torch.set_grad_enabled(False):
                next_logy = torch.empty_like(labels['logy_for_calc']).to(tu.device)
                next_logy.copy_(labels['logy_for_calc'])
                # next_logy = torch.exp(next_logy) - 1.
                if is_train:
                    random_setting = torch.empty_like(next_logy).to(tu.device).uniform_()
                    flip_mask = random_setting < c.random_flip
                    next_logy[flip_mask] = -next_logy[flip_mask]

                    sampling_mask = (random_setting >= c.random_flip) & (
                                random_setting < (c.random_flip + c.random_label))
                    next_logy[sampling_mask] = labels['mu_for_calc'][sampling_mask] + \
                                               (torch.randn_like(next_logy) * labels['sig_for_calc'])[sampling_mask]

                # next_y = next_logy
                next_y = torch.exp(next_logy) - 1.

            losses_dict = OrderedDict()
            # losses_dict['y_pf'] = -(x * labels['logy']).sum()
            losses_dict['y_pf'] = -((x - features['wgt']) * next_y).sum()
            losses_dict['mdd_pf'] = F.elu(-(x * next_y).sum(dim=1) - 0.05, 1e-6).sum()
            # losses_dict['mdd_pf'] = torch.relu(-(x * next_y).sum(dim=1) - 0.05).sum()
            losses_dict['logy'] = self.loss_func_logy(pred_mu, labels['logy'])
            losses_dict['wgt'] = nn.KLDivLoss(reduction='sum')(torch.log(x), labels['wgt'])
            losses_dict['wgt2'] = nn.KLDivLoss(reduction='sum')(torch.log(x), features['wgt'])
            losses_dict['wgt_guide'] = nn.KLDivLoss(reduction='sum')(torch.log(x), guide_wgt)

            if labels_prev is not None:
                next_y_prev = torch.exp(labels_prev['logy_for_calc']) - 1.
                wgt_prev = prev_x * (1 + next_y_prev)
                wgt_prev = wgt_prev / wgt_prev.sum(dim=1, keepdim=True)
                losses_dict['cost'] = torch.abs(x - wgt_prev).sum().exp() * c.cost_rate
            else:
                losses_dict['cost'] = torch.abs(x - features['wgt']).sum().exp() * c.cost_rate

            if losses_wgt_fixed is not None:
                if c.max_entropy:
                    losses_dict['entropy'] = -dist.entropy().sum()
                else:
                    losses_dict['entropy'] = dist.entropy().sum()
                # losses_dict['entropy'] = HLoss()(x)
                i_dict = 0
                for key in losses_dict.keys():
                    if losses_wgt_fixed[key] == 0:
                        continue

                    if i_dict == 0:
                        losses = losses_dict[key] * losses_wgt_fixed[key]
                    else:
                        losses += losses_dict[key] * losses_wgt_fixed[key]
                    i_dict += 1
            else:
                losses = losses_dict['y_pf'] + losses_dict['wgt2']
        else:
            losses = None
            losses_dict = None

        return x, losses, pred_mu, pred_sigma, losses_dict

    def save_to_optim(self):
        self.optim_state_dict = self.state_dict()

    def load_from_optim(self):
        self.load_state_dict(self.optim_state_dict)


def save_model(path, ep, model, optimizer):
    save_path = os.path.join(path, "saved_model.pt")
    torch.save({
        'ep': ep,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print('model saved successfully. ({})'.format(path))


def load_model(path, model, optimizer):
    load_path = os.path.join(path, "saved_model.pt")
    if not os.path.exists(load_path):
        return False

    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(tu.device)
    model.eval()
    print('model loaded successfully. ({})'.format(path))
    return checkpoint['ep']


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """

    def forward(self, x, gammas, betas):
        if gammas is not None:
            gammas = gammas.expand_as(x)
            x = x * gammas

        if betas is not None:
            betas = betas.expand_as(x)
            x = x + betas
        return x


class LogUniform(distributions.TransformedDistribution):
    def __init__(self, lb, ub):
        super(LogUniform, self).__init__(distributions.Uniform(lb.log(), ub.log()),
                                         distributions.ExpTransform())


class ConditionNetwork(Module):
    def __init__(self, configs):
        super(ConditionNetwork, self).__init__()
        c = configs
        self.c = c

        self.in_dim = len(c.loss_wgt)
        self.loss_dist = LogUniform(torch.tensor(1e-3), torch.tensor(1.))
        self.hidden_layers = nn.ModuleList()

        h_in = self.in_dim
        for h_out in c.film_hidden_dim:
            self.hidden_layers.append(layer.Linear(h_in, h_out))
            h_in = h_out

        self.out_layer = layer.Linear(h_in, 2 * (np.sum(c.hidden_dim) + np.sum(c.alloc_hidden_dim)))

    def forward(self, n_batch):
        c = self.c
        loss_wgt = self.loss_dist.rsample([n_batch, self.in_dim]).to(tu.device)
        x = loss_wgt[:]
        for h_layer in self.hidden_layers:
            x = h_layer(x)
            x = torch.relu(x)

        x = self.out_layer(x)
        est_gamma, est_beta = torch.chunk(x[:, :np.sum(c.hidden_dim) * 2], 2, dim=-1)
        alloc_gamma, alloc_beta = torch.chunk(x[:, np.sum(c.hidden_dim) * 2:], 2, dim=-1)

        return loss_wgt, est_gamma, est_beta, alloc_gamma, alloc_beta


class MyModel_film(Module):
    def __init__(self, in_dim, out_dim, configs, ):
        super(MyModel, self).__init__()
        c = configs
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.c = c

        self.condition_network = ConditionNetwork(configs)

        self.expected_return_estimator = ExpectedReturnEstimator(in_dim, out_dim, configs)

        self.strategies_allocator = StrategiesAllocator(in_dim + out_dim * 3, (out_dim-1) * 2, configs)

        self.optim_state_dict = self.state_dict()

    def random_guide_weight(self, features, is_train, random_guide_weight):
        c = self.c

        # guide_wgt = self.guide_weight.repeat(len(x), 1)
        # if is_train and self.random_guide_weight > 0:
        #     replace_idx = np.random.choice(np.arange(len(x)), int(len(x) * self.random_guide_weight), replace=False)
        #     guide_wgt[replace_idx] = torch.rand(len(replace_idx), guide_wgt.shape[1])
        # guide_wgt = guide_wgt.to(x.device)

        if c.base_weight is not None:
            self.guide_weight = torch.FloatTensor([c.base_weight])
        else:
            self.guide_weight = torch.ones(1, self.out_dim, dtype=torch.float32) / self.out_dim

        with torch.set_grad_enabled(False):
            if is_train:
                if np.random.rand() > random_guide_weight:
                    guide_wgt = self.guide_weight.repeat(len(features['wgt']), 1).to(features['wgt'].device)
                else:
                    guide_wgt = torch.rand_like(features['wgt']).to(features['wgt'].device)
                    guide_wgt = guide_wgt / guide_wgt.sum(dim=1, keepdim=True)
            else:
                guide_wgt = self.guide_weight.repeat(len(features['wgt']), 1).to(features['wgt'].device)

        return guide_wgt

    def forward(self, features, wgt, mc_samples):
        """
        wgt = features['wgt']
        """
        # expected return estimation
        c = self.c
        x = torch.cat([features['idx'], features['macro']], dim=-1)
        n_batch = x.shape[0]
        if c.use_condition_network:
            losses_wgt, est_gamma, est_beta, alloc_gamma, alloc_beta = self.condition_network(n_batch)
        else:
            losses_wgt = torch.ones(n_batch, c.loss_wgt).to(tu.device)
            est_gamma, est_beta, alloc_gamma, alloc_beta = None, None, None, None

        pred, pred_mu, pred_sigma = self.expected_return_estimator.run(x, mc_samples=mc_samples, film_gamma=est_gamma, film_beta=est_beta)

        # allocation
        x = torch.cat([pred_mu, pred_sigma, wgt, x], dim=-1)
        wgt_mu, wgt_sigma = self.strategies_allocator(x, film_gamma=alloc_gamma, film_beta=alloc_beta)

        return wgt_mu, wgt_sigma, pred_mu, pred_sigma, losses_wgt

    def predict(self, features, features_prev, is_train, mc_samples):
        c = self.c
        guide_wgt = self.random_guide_weight(features, is_train, c.random_guide_weight)

        if features_prev is not None:
            with torch.set_grad_enabled(False):
                # prev_x, _, _ = self.run(features_prev, features['wgt'], n_samples)
                prev_wgt, _, _, _, _ = self.forward(features_prev, guide_wgt, mc_samples)
                prev_wgt = (1. + prev_wgt) * guide_wgt
                prev_wgt = prev_wgt / (prev_wgt.sum(-1, keepdim=True) + 1e-6)
        else:
            prev_wgt = guide_wgt

        # x, pred_mu, pred_sigma = self.run(features, prev_x, n_samples)
        if c.use_guide_wgt_as_prev_x is True:
            wgt_mu, wgt_sigma, pred_mu, pred_sigma, losses_wgt = self.forward(features, guide_wgt, mc_samples)
        else:
            wgt_mu, wgt_sigma, pred_mu, pred_sigma, losses_wgt = self.forward(features, prev_wgt, mc_samples)

        dist = torch.distributions.Normal(loc=wgt_mu, scale=wgt_sigma)
        if is_train:
            wgt_ = dist.rsample().clamp(0.01, 0.99)
            wgt_ = wgt_ / (wgt_.sum(-1, keepdim=True) + 1e-6)
        else:
            wgt_ = wgt_mu

        # cash 제한 풀기
        noncash_idx = np.delete(np.arange(guide_wgt.shape[1]), c.cash_idx)
        wgt_ = (1. + wgt_) * guide_wgt[:, noncash_idx]

        wgt_rf = torch.max(1 - wgt_.sum(-1, keepdim=True),
                              torch.zeros(wgt_.shape[0], 1, device=wgt_.device))
        wgt_ = torch.cat([wgt_, wgt_rf], dim=-1)

        wgt_ = wgt_ / (wgt_.sum(-1, keepdim=True) + 1e-6)
        return dist, wgt_, prev_wgt, pred_mu, pred_sigma, guide_wgt, losses_wgt

    @profile
    def forward_with_loss(self, features, labels=None,
                          mc_samples=1000,
                          losses_wgt_fixed=None,
                          features_prev=None,
                          labels_prev=None,
                          is_train=True,
                          ):
        """
        t = 2000; mc_samples = 200; is_train = True
        dataset = {'train': None, 'eval': None, 'test': None, 'test_insample': None}
        dataset['train'], dataset['eval'], dataset['test'], (dataset['test_insample'], insample_boundary), guide_date = sampler.get_batch(t)

        dataset['train'], train_n_samples = dataset['train']

        dataset['train'] = tu.to_device(tu.device, to_torch(dataset['train']))
        train_features_prev, train_labels_prev, train_features, train_labels = dataset['train']
        features_prev, labels_prev, features, labels = dict(), dict(), dict(), dict()
        sampled_batch_idx = np.random.choice(np.arange(train_n_samples), c.batch_size)
        for key in train_features.keys():
            features_prev[key] = train_features_prev[key][sampled_batch_idx]
            features[key] = train_features[key][sampled_batch_idx]

        for key in train_labels.keys():
            labels_prev[key] = train_labels_prev[key][sampled_batch_idx]
            labels[key] = train_labels[key][sampled_batch_idx]


        """
        c = self.c
        dist, x, prev_x, pred_mu, pred_sigma, guide_wgt, losses_wgt = self.predict(features, features_prev, is_train, mc_samples)

        if torch.isnan(x).sum() > 0:
            for n, p in list(self.named_parameters()):
                print(n, '\nval:\n', p, '\ngrad:\n', p.grad)

            return False

        if labels is not None:
            with torch.set_grad_enabled(False):
                next_logy = torch.empty_like(labels['logy_for_calc']).to(tu.device)
                next_logy.copy_(labels['logy_for_calc'])
                # next_logy = torch.exp(next_logy) - 1.
                if is_train:
                    random_setting = torch.empty_like(next_logy).to(tu.device).uniform_()
                    flip_mask = random_setting < c.random_flip
                    next_logy[flip_mask] = -next_logy[flip_mask]

                    sampling_mask = (random_setting >= c.random_flip) & (
                                random_setting < (c.random_flip + c.random_label))
                    next_logy[sampling_mask] = labels['mu_for_calc'][sampling_mask] + \
                                               (torch.randn_like(next_logy) * labels['sig_for_calc'])[sampling_mask]

                # next_y = next_logy
                next_y = torch.exp(next_logy) - 1.

            losses_dict = dict()
            losses_wgt_dict = dict()
            # losses_dict['y_pf'] = -(x * labels['logy']).sum()
            if c.use_condition_network:
                for key in c.loss_list:
                    losses_wgt_dict[key] = losses_wgt[:, c.loss_list.index(key)].unsqueeze(-1)
            else:
                losses_wgt_dict = losses_wgt_fixed

            losses_dict['y_pf'] = -((x - features['wgt']) * next_y * losses_wgt_dict['y_pf']).sum()
            losses_dict['mdd_pf'] = (F.elu(-(x * next_y).sum(dim=1) - 0.05, 1e-6) * losses_wgt_dict['mdd_pf']).sum()
            # losses_dict['mdd_pf'] = torch.relu(-(x * next_y).sum(dim=1) - 0.05).sum()
            losses_dict['logy'] = (nn.MSELoss(reduction='none')(pred_mu, labels['logy']) * losses_wgt_dict['logy']).sum()
            losses_dict['wgt_guide'] = (nn.KLDivLoss(reduction='none')(torch.log(x), guide_wgt) * losses_wgt_dict['wgt_guide']).sum()

            if labels_prev is not None:
                next_y_prev = torch.exp(labels_prev['logy_for_calc']) - 1.
                wgt_prev = prev_x * (1 + next_y_prev)
                wgt_prev = wgt_prev / wgt_prev.sum(dim=1, keepdim=True)
                losses_dict['cost'] = (torch.abs(x - wgt_prev) * c.cost_rate * losses_wgt_dict['cost']).sum()
            else:
                losses_dict['cost'] = (torch.abs(x - features['wgt']) * c.cost_rate * losses_wgt_dict['cost']).sum()

            if c.max_entropy:
                losses_dict['entropy'] = (-dist.entropy() * losses_wgt_dict['entropy']).sum()
            else:
                losses_dict['entropy'] = (dist.entropy() * losses_wgt_dict['entropy']).sum()
            # losses_dict['entropy'] = HLoss()(x)

            for i_dict, key in enumerate(losses_dict.keys()):
                if i_dict == 0:
                    losses = losses_dict[key]
                else:
                    losses += losses_dict[key]
        else:
            losses = None
            losses_dict = None

        return x, losses, pred_mu, pred_sigma, losses_dict

    def adversarial_noise(self, features, labels, features_prev=None, labels_prev=None):
        for key in features.keys():
            features[key].requires_grad = True

        for key in labels.keys():
            labels[key].requires_grad = True

        pred, losses, _, _, _ = self.forward_with_loss(features, labels, mc_samples=100, losses_wgt_fixed=None,
                                                       features_prev=features_prev, labels_prev=labels_prev)

        losses.backward(retain_graph=True)
        features_grad = dict()
        features_perturbed = dict()
        for key in features.keys():
            features_grad[key] = torch.sign(features[key].grad)
            sample_sigma = torch.std(features[key] + 1e-6, axis=[0], keepdims=True)
            eps = 0.05
            scaled_eps = eps * sample_sigma  # [1, 1, n_features]

            features_perturbed[key] = features[key] + scaled_eps * features_grad[key]
            features[key].grad.zero_()

        return features_perturbed

    def save_to_optim(self):
        self.optim_state_dict = self.state_dict()

    def load_from_optim(self):
        self.load_state_dict(self.optim_state_dict)


