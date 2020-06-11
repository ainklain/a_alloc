import os
import numpy as np
import torch
from torch import nn
from torch.nn import init, Module, functional as F
from torch.nn.modules import Linear
from torch.distributions import Categorical

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


class MyModel(Module):
    def __init__(self, in_dim, out_dim, configs,):
        super(MyModel, self).__init__()
        c = configs
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.c = c
        self.cost_rate = c.cost_rate
        self.dropout_r = c.dropout_r
        self.max_entropy = c.max_entropy
        self.random_guide_weight = c.random_guide_weight
        self.random_label = c.random_label

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
        mask = self.training or sample
        n_samples, batch_size, _ = x.shape
        for h_layer, bn in zip(self.hidden_layers, self.hidden_bn):
            x = h_layer(x)

            # x = bn(x.view(-1, bn.num_features)).view(n_samples, batch_size, bn.num_features)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout_r, training=mask)

        x = self.out_layer(x)
        # print([p.grad for p in list(self.out_layer.parameters()) if p.grad is not None])
        return x

    def sample_predict(self, x, mc_samples):
        """
        x = torch.randn(512, 30)
        """
        self.train()
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
        # x = self.aa_bn(x)
        x = F.relu(x)
        x = self.aa_hidden_layer2(x)
        # x = self.aa_bn2(x)
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
                if np.random.rand() > self.random_guide_weight:
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
            next_y = torch.exp(labels['logy_for_calc']) - 1.

            with torch.set_grad_enabled(False):
                if is_train:
                    mask = torch.empty_like(labels['logy_for_calc']).to(tu.device).uniform_() < self.random_label
                    next_y[mask] = -next_y[mask]

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
                losses_dict['cost'] = torch.abs(x - wgt_prev).sum() * self.cost_rate
            else:
                losses_dict['cost'] = torch.abs(x - features['wgt']).sum() * self.cost_rate

            if loss_wgt is not None:
                if self.max_entropy:
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
    print('model saved successfully.')


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
    print('model loaded successfully.')
    return checkpoint['ep']





# class Model(Module):
#     def __init__(self, in_dim, out_dim, hidden_dim=[72, 48, 32], dropout_r=0.5):
#         super(Model, self).__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.dropout_r = dropout_r
#         self.hidden_layers = nn.ModuleList()
#
#         h_in = in_dim
#         for h_out in hidden_dim:
#             self.hidden_layers.append(XLinear(h_in, h_out))
#             h_in = h_out
#
#         self.out_layer = XLinear(h_in, out_dim)
#
#         # asset allocation
#         self.aa_hidden_layer = XLinear(out_dim * 3, 4)
#         self.aa_out_layer = XLinear(4, out_dim)
#
#     def forward_notused(self, x, sample=True):
#         mask = self.training or sample
#
#         for h_layer in self.hidden_layers:
#             x = h_layer(x)
#             x = F.dropout(x, p=self.dropout_r, training=mask)
#             x = F.relu(x)
#
#         x = self.out_layer(x)
#         return x
#
#     def sample_predict_notused(self, x, n_samples):
#         # Just copies type from x, initializes new vector
#         predictions = x.data.new(n_samples, x.shape[0], self.out_dim)
#
#         for i in range(n_samples):
#             y = self.forward(x, sample=True)
#             predictions[i] = y
#
#         return predictions
#
#     def forward_with_correction_notused(self, features, labels=None, n_samples=1000):
#         pred = self.sample_predict(features, n_samples=n_samples)
#         pred_mu = torch.mean(pred, dim=0)
#         pred_sigma = torch.std(pred, dim=0)
#         if labels is not None:
#             error_correction = torch.zeros_like(pred_mu) + torch.abs(labels - pred_mu)
#         else:
#             error_correction = torch.zeros_like(pred_mu)
#
#         x = torch.cat([pred_mu, pred_sigma, error_correction], dim=-1)
#         x = self.aa_hidden_layer(x)
#         x = F.relu(x)
#         x = self.aa_out_layer(x)
#         x = F.softmax(x)
#
#         return x
