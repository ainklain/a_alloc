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
    def __init__(self, in_dim, out_dim, hidden_dim=[72, 48, 32], dropout_r=0.5, cost_rate=0.003):
        super(MyModel, self).__init__()
        self.cost_rate = cost_rate
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout_r = dropout_r
        self.hidden_layers = nn.ModuleList()

        h_in = in_dim
        for h_out in hidden_dim:
            self.hidden_layers.append(XLinear(h_in, h_out))
            h_in = h_out

        self.out_layer = XLinear(h_in, out_dim)

        # asset allocation
        self.aa_hidden_layer = XLinear(out_dim * 3, out_dim)
        self.aa_out_layer = XLinear(out_dim, out_dim)

        self.loss_func_logy = nn.MSELoss()

        self.optim_state_dict = self.state_dict()
        # self.optim_
    # def init_weight(self):
    #     return torch.ones_like()

    def forward(self, x, sample=True):
        mask = self.training or sample

        for h_layer in self.hidden_layers:
            x = h_layer(x)
            x = F.dropout(x, p=self.dropout_r, training=mask)
            x = F.relu(x)

        x = self.out_layer(x)
        return x

    def sample_predict(self, x, n_samples):
        self.train()
        # Just copies type from x, initializes new vector
        # predictions = x.data.new(n_samples, x.shape[0], self.out_dim)
        #
        # for i in range(n_samples):
        #     y = self.forward(x, sample=True)
        #     predictions[i] = y

        predictions = x.unsqueeze(0).repeat(n_samples, 1, 1)
        predictions = self.forward(predictions, sample=True)
        return predictions

    def adversarial_noise(self, features, labels):
        for key in features.keys():
            features[key].requires_grad = True

        for key in labels.keys():
            labels[key].requires_grad = True

        pred, losses, _, _, _ = self.forward_with_loss(features, labels, n_samples=100, loss_wgt=None)

        losses.backward(retain_graph=True)
        features_grad = dict()
        features_perturbed = dict()
        for key in features.keys():
            features_grad[key] = torch.sign(features[key].grad)
            sample_sigma = torch.std(features[key], axis=[0], keepdims=True)
            eps = 0.01
            scaled_eps = eps * sample_sigma  # [1, 1, n_features]

            features_perturbed[key] = features[key] + scaled_eps * features_grad[key]
            features[key].grad.zero_()

        return features_perturbed

    @profile
    def forward_with_loss(self, features, labels=None, n_samples=1000, loss_wgt=None):
        # features, labels=  train_features, train_labels
        x = torch.cat([features['idx'], features['macro']], dim=-1)
        pred = self.sample_predict(x, n_samples=n_samples)
        pred_mu = torch.mean(pred, dim=0)
        pred_sigma = torch.std(pred, dim=0)

        # if labels is not None:
        #     error_correction = torch.zeros_like(pred_mu) + torch.abs(labels['logy'] - pred_mu)
        # else:
        #     error_correction = torch.zeros_like(pred_mu)


        # rand_val = np.random.random()
        # if rand_val >= 0.5:
        #     add_info = features['wgt']
        # elif rand_val >= 0.1:
        #     add_info = guide_wgt
        # else:
        #     add_info = torch.rand_like(guide_wgt)
        #     add_info = add_info / add_info.sum(dim=1, keepdim=True)
        # x = torch.cat([pred_mu, pred_sigma, add_info], dim=-1)

        min_wgt = torch.zeros_like(features['wgt'], dtype=torch.float32).to(x.device) + 0.001

        x = torch.cat([pred_mu, pred_sigma, features['wgt']], dim=-1)
        x = self.aa_hidden_layer(x)
        x = F.relu(x)
        x = self.aa_out_layer(x)
        x = F.elu(x + features['wgt'], 0.01) + min_wgt + 0.01
        # x = F.softmax(x)
        x = x / x.sum(dim=1, keepdim=True)

        guide_wgt = torch.FloatTensor([[0.699, 0.2, 0.1, 0.001]]).repeat(len(x), 1).to(x.device)

        if labels is not None:
            losses_dict = dict()
            # losses_dict['logy_pf'] = -(x * labels['logy']).sum()
            losses_dict['logy_pf'] = -((x - features['wgt']) * labels['logy']).sum()
            losses_dict['mdd_pf'] = F.elu(-(x * labels['logy']).sum(dim=1) - 0.05, 0.001).sum()
            losses_dict['logy'] = self.loss_func_logy(pred_mu, labels['logy'])
            losses_dict['wgt'] = nn.KLDivLoss(reduction='sum')(torch.log(x), labels['wgt'])
            losses_dict['wgt2'] = nn.KLDivLoss(reduction='sum')(torch.log(x), features['wgt'])
            losses_dict['wgt_guide'] = nn.KLDivLoss(reduction='sum')(torch.log(x), guide_wgt)
            losses_dict['cost'] = torch.abs(x - features['wgt']).sum() * self.cost_rate

            if loss_wgt is not None:
                losses_dict['entropy'] = HLoss()(x)
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
                losses = losses_dict['logy_pf'] + losses_dict['wgt2']
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
