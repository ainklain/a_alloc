import os
import numpy as np
import torch
from torch import nn
from torch.nn import init, Module, functional as F
from torch.nn.modules import Linear
from torch.distributions import Categorical
import torch.distributions as dist

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


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        # b = -1.0 * b.sum(dim=1)
        # b = b.mean()
        return b


class XLinear(Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(XLinear, self).__init__()
        self.layer = Linear(in_dim, out_dim, bias)
        init.xavier_uniform_(self.layer.weight)
        if bias:
            init.zeros_(self.layer.bias)

    def forward(self, x):
        return self.layer(x)


class FiLM(nn.Module):
  """
  A Feature-wise Linear Modulation Layer from
  'FiLM: Visual Reasoning with a General Conditioning Layer'
  """
  def forward(self, x, gammas, betas):
    gammas = gammas.expand_as(x)
    betas = betas.expand_as(x)
    return (gammas * x) + betas


class LogUniform(dist.TransformedDistribution):
    def __init__(self, lb, ub):
        super(LogUniform, self).__init__(dist.Uniform(lb.log(), ub.log()),
                                         dist.ExpTransform())


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
            self.hidden_layers.append(XLinear(h_in, h_out))
            h_in = h_out

        self.out_layer = XLinear(h_in, 2 * (np.sum(c.hidden_dim) + np.sum(c.alloc_hidden_dim)))

    def forward(self, n_batch):
        c = self.c
        x = self.loss_dist.rsample([n_batch, self.in_dim])
        for h_layer in self.hidden_layers:
            x = h_layer(x)
            x = torch.relu(x)

        x = self.out_layer(x)
        est_gamma, est_beta = torch.chunk(x[:, :np.sum(c.hidden_dim) * 2], 2, dim=-1)
        alloc_gamma, alloc_beta = torch.chunk(x[:, np.sum(c.hidden_dim) * 2:], 2, dim=-1)

        return x, est_gamma, est_beta, alloc_gamma, alloc_beta


class ExpectedReturnEstimator(Module):
    def __init__(self, in_dim, out_dim, configs):
        super(ExpectedReturnEstimator, self).__init__()

        c = configs
        self.c = c

        if c.use_condition_network:
            self.film = FiLM()

        self.hidden_layers = nn.ModuleList()
        self.hidden_bn = nn.ModuleList()

        h_in = in_dim
        for h_out in c.hidden_dim:
            self.hidden_layers.append(XLinear(h_in, h_out))
            self.hidden_bn.append(nn.BatchNorm1d(h_out))
            h_in = h_out

        self.out_layer = XLinear(h_out, out_dim)

    def forward(self, x, sample=True, film_gamma=None, film_beta=None):
        """
        x = torch.randn(200, 512, 30).cuda()
        """
        c = self.c
        mask = self.training or sample
        n_samples, batch_size, _ = x.shape
        for i in range(len(self.hidden_layers)):
            h_layer, bn = self.hidden_layers[i], self.hidden_bn[i]
            if film_gamma is not None:
                if i == 0:
                    gamma = film_gamma[:, :c.hidden_dim[i]]
                else:
                    gamma = film_gamma[:, c.hidden_dim[i-1]:c.hidden_dim[i]]
                gamma = gamma.unsqueeze(0) #.to(tu.device)

            if film_beta is not None:
                if i == 0:
                    beta = film_beta[:, :c.hidden_dim[i]]
                else:
                    beta = film_beta[:, c.hidden_dim[i-1]:c.hidden_dim[i]]
                beta = beta.unsqueeze(0) #.to(tu.device)

            x = h_layer(x)
            x = bn(x.view(-1, bn.num_features)).view(n_samples, batch_size, bn.num_features)
            if c.use_condition_network:
                x = self.film(x, gamma, beta)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=c.dropout_r, training=mask)

        x = self.out_layer(x)
        # print([p.grad for p in list(self.out_layer.parameters()) if p.grad is not None])
        return x

    def run(self, x, mc_samples):
        """
        x = torch.cat([features['idx'], features['macro']], dim=-1)
        x = torch.randn(512, 30).cuda()
        """
        x = x.unsqueeze(0).repeat(mc_samples, 1, 1)
        x = self.forward(x, sample=True)

        pred = self.sample_predict(x, mc_samples=mc_samples)
        pred_mu = torch.mean(pred, dim=0)
        with torch.set_grad_enabled(False):
            pred_sigma = torch.std(pred + 1e-6, dim=0, unbiased=False)

        return pred, pred_mu, pred_sigma


class StrategiesAllocator(Module):
    def __init__(self, in_dim, out_dim, configs):
        """
        :param in_dim: pred_mu + pred_sigma + prev_wgt + input =
        :param out_dim: wgt_mu + wgt_sigma
        """
        c = configs
        self.c = c

        if c.use_condition_network:
            self.film = FiLM()

        self.hidden_layers = nn.ModuleList()
        self.hidden_bn = nn.ModuleList()

        h_in = in_dim
        for h_out in c.alloc_hidden_dim:
            self.hidden_layers.append(XLinear(h_in, h_out))
            self.hidden_bn.append(nn.BatchNorm1d(h_out))
            h_in = h_out

        self.out_layer = XLinear(h_out, out_dim)

    def forward(self, x, film_gamma=None, film_beta=None):
        c = self.c
        for i in range(len(self.hidden_layers)):
            h_layer, bn = self.hidden_layers[i], self.hidden_bn[i]
            if film_gamma is not None:
                if i == 0:
                    gamma = film_gamma[:, :c.hidden_dim[i]]
                else:
                    gamma = film_gamma[:, c.hidden_dim[i-1]:c.hidden_dim[i]]

            if film_beta is not None:
                if i == 0:
                    beta = film_beta[:, :c.hidden_dim[i]]
                else:
                    beta = film_beta[:, c.hidden_dim[i-1]:c.hidden_dim[i]]
            x = h_layer(x)
            x = bn(x)
            if c.use_condition_network:
                x = self.film(x, gamma, beta)
            x = F.relu(x)

        x = self.out_layer(x)
        wgt_mu, wgt_logsigma = torch.chunk(x, 2, dim=-1)
        wgt_mu = 0.99 * torch.tanh(wgt_mu) + 1e-6
        wgt_sigma = 0.2 * F.softplus(wgt_logsigma) + 1e-6

        return wgt_mu, wgt_sigma


class MyModel(Module):
    def __init__(self, in_dim, out_dim, configs,):
        super(MyModel, self).__init__()
        c = configs
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.c = c

        self.condition_network = ConditionNetwork(configs)

        self.expected_return_estimator = ExpectedReturnEstimator(in_dim, out_dim, configs)

        self.strategies_allocator = StrategiesAllocator(in_dim + out_dim * 3, out_dim * 2, configs)

        self.loss_func_logy = nn.MSELoss()

        self.optim_state_dict = self.state_dict()

    def random_guide_weight(self, features, is_train, random_guide_weight):
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
        x = torch.cat([features['idx'], features['macro']], dim=-1)
        pred, pred_mu, pred_sigma = self.expected_return_estimator(x, mc_samples=mc_samples)

        # allocation
        x = torch.cat([pred_mu, pred_sigma, wgt, x], dim=-1)
        wgt_mu, wgt_sigma = self.strategies_allocator(x)

        return wgt_mu, wgt_sigma, pred_mu, pred_sigma

    def predict(self, features, features_prev, is_train, mc_samples):
        c = self.c
        guide_wgt = self.random_guide_weight(self, features, is_train, c.random_guide_weight)

        if features_prev is not None:
            with torch.set_grad_enabled(False):
                # prev_x, _, _ = self.run(features_prev, features['wgt'], n_samples)
                prev_x, _, _, _ = self.forward(features_prev, guide_wgt, mc_samples)
                prev_x = (1. + prev_x) * guide_wgt
                prev_x = prev_x / (prev_x.sum(-1, keepdim=True) + 1e-6)
        else:
            prev_x = guide_wgt

        # x, pred_mu, pred_sigma = self.run(features, prev_x, n_samples)
        if c.use_guide_wgt_as_prev_x is True:
            wgt_mu, wgt_sigma, pred_mu, pred_sigma = self.forward(features, guide_wgt, mc_samples)
        else:
            wgt_mu, wgt_sigma, pred_mu, pred_sigma = self.forward(features, prev_x, mc_samples)

        wgt_mu = (1. + wgt_mu) * guide_wgt

        # cash 제한 풀기
        noncash_idx = np.delete(np.arange(wgt_mu.shape[1]), c.cash_idx)
        wgt_mu_rf = torch.max(1 - wgt_mu[:, noncash_idx].sum(-1, keepdim=True), torch.zeros(wgt_mu.shape[0], 1, device=wgt_mu.device))
        wgt_mu = torch.cat([wgt_mu[:, noncash_idx], wgt_mu_rf], dim=-1)

        wgt_mu = wgt_mu / (wgt_mu.sum(-1, keepdim=True) + 1e-6)
        dist = torch.distributions.Normal(loc=wgt_mu, scale=wgt_sigma)
        if is_train:
            x = dist.rsample().clamp(0.01, 0.99)
            x = x / (x.sum(-1, keepdim=True) + 1e-6)
        else:
            x = wgt_mu

        return dist, x, prev_x, pred_mu, pred_sigma, guide_wgt

    @profile
    def forward_with_loss(self, features, labels=None,
                          mc_samples=1000,
                          loss_wgt=None,
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
        dist, x, prev_x, pred_mu, pred_sigma, guide_wgt = self.predict(features, features_prev, is_train, mc_samples)

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

                    sampling_mask = (random_setting >= c.random_flip) & (random_setting < (c.random_flip+c.random_label))
                    next_logy[sampling_mask] = labels['mu_for_calc'][sampling_mask] + (torch.randn_like(next_logy) * labels['sig_for_calc'])[sampling_mask]

                # next_y = next_logy
                next_y = torch.exp(next_logy) - 1.

            losses_dict = dict()
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
                wgt_prev = prev_x * (1+next_y_prev)
                wgt_prev = wgt_prev / wgt_prev.sum(dim=1, keepdim=True)
                losses_dict['cost'] = torch.abs(x - wgt_prev).sum() * c.cost_rate
            else:
                losses_dict['cost'] = torch.abs(x - features['wgt']).sum() * c.cost_rate

            if loss_wgt is not None:
                if c.max_entropy:
                    losses_dict['entropy'] = -dist.entropy().sum()
                else:
                    losses_dict['entropy'] = dist.entropy().sum()
                # losses_dict['entropy'] = HLoss()(x)
                i_dict = 0
                for key in losses_dict.keys():
                    if loss_wgt[key] == 0:
                        continue

                    if i_dict == 0:
                        losses = losses_dict[key] * loss_wgt[key]
                    else:
                        losses += losses_dict[key] * loss_wgt[key]
                    i_dict += 1
            else:
                losses = losses_dict['y_pf'] + losses_dict['wgt2']
        else:
            losses = None
            losses_dict = None

        return x, losses, pred_mu, pred_sigma, losses_dict

    def adversarial_noise(self, features, labels, features_prev=None, labels_prev=None):
        for key in features.keys():
            features[key].requires_grad = True

        for key in labels.keys():
            labels[key].requires_grad = True

        pred, losses, _, _, _ = self.forward_with_loss(features, labels, mc_samples=100, loss_wgt=None,
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



class MyModel_old(Module):
    def __init__(self, in_dim, out_dim, configs,):
        super(MyModel, self).__init__()
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
            self.hidden_layers.append(XLinear(h_in, h_out))
            self.hidden_bn.append(nn.BatchNorm1d(h_out))
            h_in = h_out

        self.out_layer = XLinear(h_out, out_dim)

        # asset allocation
        self.aa_hidden_layer = XLinear(out_dim * 3 + in_dim, (out_dim * 3 + in_dim)//2)
        self.aa_hidden_layer2 = XLinear((out_dim * 3 + in_dim) // 2, out_dim * 2)
        self.aa_bn = nn.BatchNorm1d((out_dim * 3 + in_dim)//2)
        self.aa_bn2 = nn.BatchNorm1d(out_dim * 2)

        self.aa_out_layer = XLinear(out_dim * 2, out_dim * 2)

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

        pred, losses, _, _, _ = self.forward_with_loss(features, labels, mc_samples=100, loss_wgt=None,
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
                          loss_wgt=None,
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
        wgt_mu_rf = torch.max(1 - wgt_mu[:, noncash_idx].sum(-1, keepdim=True), torch.zeros(wgt_mu.shape[0], 1, device=wgt_mu.device))
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

                    sampling_mask = (random_setting >= c.random_flip) & (random_setting < (c.random_flip+c.random_label))
                    next_logy[sampling_mask] = labels['mu_for_calc'][sampling_mask] + (torch.randn_like(next_logy) * labels['sig_for_calc'])[sampling_mask]

                # next_y = next_logy
                next_y = torch.exp(next_logy) - 1.

            losses_dict = dict()
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
                wgt_prev = prev_x * (1+next_y_prev)
                wgt_prev = wgt_prev / wgt_prev.sum(dim=1, keepdim=True)
                losses_dict['cost'] = torch.abs(x - wgt_prev).sum() * c.cost_rate
            else:
                losses_dict['cost'] = torch.abs(x - features['wgt']).sum() * c.cost_rate

            if loss_wgt is not None:
                if c.max_entropy:
                    losses_dict['entropy'] = -dist.entropy().sum()
                else:
                    losses_dict['entropy'] = dist.entropy().sum()
                # losses_dict['entropy'] = HLoss()(x)
                i_dict = 0
                for key in losses_dict.keys():
                    if loss_wgt[key] == 0:
                        continue

                    if i_dict == 0:
                        losses = losses_dict[key] * loss_wgt[key]
                    else:
                        losses += losses_dict[key] * loss_wgt[key]
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


