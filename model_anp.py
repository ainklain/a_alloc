import os
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init, Module, functional as F
# from torch.nn.modules import Linear
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


class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        # Linear
        return self.linear_layer(x)


class XLinear(Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(XLinear, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim, bias)
        init.xavier_uniform_(self.layer.weight)
        if bias:
            init.zeros_(self.layer.bias)

    def forward(self, x):
        return self.layer(x)


class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism (dot attention)
    """

    def __init__(self, num_hidden_k):
        """
        :param num_hidden_k: dimension of hidden
        """
        super(MultiheadAttention, self).__init__()

        self.num_hidden_k = num_hidden_k
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query):
        # MHA
        # Get attention score
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / math.sqrt(self.num_hidden_k)

        attn = torch.softmax(attn, dim=-1)

        # Dropout
        attn = self.attn_dropout(attn)

        # Get Context Vector
        result = torch.bmm(attn, value)

        return result, attn


class Attention(nn.Module):
    """
    Attention Network
    """

    def __init__(self, num_hidden, h=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        """
        super(Attention, self).__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = Linear(num_hidden, num_hidden, bias=False)
        self.value = Linear(num_hidden, num_hidden, bias=False)
        self.query = Linear(num_hidden, num_hidden, bias=False)

        self.multihead = MultiheadAttention(self.num_hidden_per_attn)

        self.residual_dropout = nn.Dropout(p=0.1)

        self.final_linear = Linear(num_hidden * 2, num_hidden)

        self.layer_norm = nn.LayerNorm(num_hidden)

    @profile
    def forward(self, key, value, query):
        # Attention

        batch_size = key.size(0)
        seq_k = key.size(1)
        seq_q = query.size(1)
        residual = query

        # Make multihead
        key = self.key(key).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        value = self.value(value).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        query = self.query(query).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)

        key = key.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        value = value.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)

        # Get context vector
        result, attns = self.multihead(key, value, query)

        # Concatenate all multihead context vector
        result = result.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        result = result.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)

        # Concatenate context vector with input (most important)
        result = torch.cat([residual, result], dim=-1)

        # Final linear
        result = self.final_linear(result)

        # Residual dropout & connection
        result = self.residual_dropout(result)
        result = result + residual

        # Layer normalization
        result = self.layer_norm(result)

        return result, attns


class LatentContext(nn.Module):
    """
    Latent Encoder [For prior, posterior]
    """

    def __init__(self, num_hidden, num_latent, input_dim=3):
        super(LatentContext, self).__init__()
        self.input_projection = Linear(input_dim, num_hidden)
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.penultimate_layer = Linear(num_hidden, num_hidden, w_init='relu')
        self.mu = Linear(num_hidden, num_latent)
        self.log_sigma = Linear(num_hidden, num_latent)

        self.local_penultimate_layer = Linear(num_hidden, num_hidden, w_init='relu')
        self.local_mu = Linear(num_hidden, num_latent)
        self.local_log_sigma = Linear(num_hidden, num_latent)

    @profile
    def forward(self, x, y):
        # LatentContext
        # concat location (x) and value (y)
        encoder_input = torch.cat([x, y], dim=-1)

        # project vector with dimension 3 --> num_hidden
        encoder_input = self.input_projection(encoder_input)

        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        # global dist
        hidden = encoder_input.mean(dim=1)
        hidden = torch.relu(self.penultimate_layer(hidden))

        global_mu = self.mu(hidden)
        global_log_sigma = self.log_sigma(hidden)
        global_sigma = torch.exp(0.5 * global_log_sigma)
        # global_dist = torch.distributions.Normal(loc=mu, scale=sigma)

        # # reparameterization trick
        # sigma = torch.exp(0.5 * log_sigma)
        # eps = torch.randn_like(sigma)
        # z = eps.mul(sigma).add_(mu)

        # local dist
        local_hidden = torch.relu(self.local_penultimate_layer(encoder_input))
        local_mu = self.local_mu(local_hidden)
        local_log_sigma = self.local_log_sigma(local_hidden)
        local_sigma = torch.exp(0.5 * local_log_sigma)
        # local_dist = torch.distributions.Normal(loc=local_mu, scale=local_sigma)

        # return distribution
        # return mu, log_sigma, z
        return global_mu, global_sigma, local_mu, local_sigma


class LatentEncoder(nn.Module):
    def __init__(self, num_hidden, num_latent, d_x=1, d_y=1):
        super(LatentEncoder, self).__init__()
        self.latent_context = LatentContext(num_hidden, num_latent, d_x+d_y)
        self.context_projection = Linear(d_x, num_hidden)
        self.target_projection = Linear(d_x, num_hidden)
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.out_layer = Linear(num_latent * 2, num_latent, w_init='relu')

    @profile
    def forward(self, context_x, context_y, target_x):
        # LatentEncoder
        num_targets = target_x.size(1)

        global_mu, global_sigma, local_mu, local_sigma = self.latent_context(context_x, context_y)
        # local
        local_eps = torch.randn_like(local_sigma)
        local_z = local_eps.mul(local_sigma).add_(local_mu)

        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        # cross attention layer
        for attention in self.cross_attentions:
            query, _ = attention(keys, local_z, query)

        # global
        global_eps = torch.randn_like(global_sigma)
        global_z = global_eps.mul(global_sigma).add_(global_mu)
        global_z = global_z.unsqueeze(1).repeat(1, num_targets, 1)  # [B, T_target, H]

        c_latent = torch.cat([query, global_z], dim=-1)
        c = torch.relu(self.out_layer(c_latent))
        return c, global_z


class DeterministicEncoder(nn.Module):
    """
    Deterministic Encoder [r]
    """

    def __init__(self, num_hidden, d_x=1, d_y=1):
        super(DeterministicEncoder, self).__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.cross_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.input_projection = Linear(d_x + d_y, num_hidden)
        self.context_projection = Linear(d_x, num_hidden)
        self.target_projection = Linear(d_x, num_hidden)

    @profile
    def forward(self, context_x, context_y, target_x):
        # DeterministicEncoder
        # concat context location (x), context value (y)
        encoder_input = torch.cat([context_x, context_y], dim=-1)

        # project vector with dimension 3 --> num_hidden
        encoder_input = self.input_projection(encoder_input)

        # self attention layer
        for attention in self.self_attentions:
            encoder_input, _ = attention(encoder_input, encoder_input, encoder_input)

        # query: target_x, key: context_x, value: representation
        query = self.target_projection(target_x)
        keys = self.context_projection(context_x)

        # cross attention layer
        for attention in self.cross_attentions:
            query, _ = attention(keys, encoder_input, query)

        return query


class Decoder(nn.Module):
    """
    Decoder for generation
    """

    def __init__(self, num_hidden, d_x=1, d_y=1):
        super(Decoder, self).__init__()
        self.self_attentions = nn.ModuleList([Attention(num_hidden) for _ in range(2)])
        self.target_projection = Linear(d_x, num_hidden)
        self.linears = nn.ModuleList([Linear(num_hidden * 4, num_hidden * 4, w_init='relu') for _ in range(3)])
        self.penultimate_layer = Linear(num_hidden * 4, num_hidden, w_init='relu')
        self.final_projection = Linear(num_hidden, d_y * 2)

    @profile
    def forward(self, r, z, c, target_x):
        # Decoder
        batch_size, num_targets, _ = target_x.size()
        # project vector with dimension 2 --> num_hidden
        target_x = self.target_projection(target_x)

        # concat all vectors (r,z,c, target_x)
        hidden = torch.cat([r, z, c, target_x], dim=-1)

        # mlp layers
        for linear in self.linears:
            hidden = torch.relu(linear(hidden))

        hidden = self.penultimate_layer(hidden)

        for attention in self.self_attentions:
            hidden, _ = attention(hidden, hidden, hidden)

        # get mu and sigma
        y_pred = self.final_projection(hidden)

        # Get the mean an the variance
        mu, log_sigma = torch.chunk(y_pred, 2, dim=-1)

        # Bound the variance
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        dist = torch.distributions.Normal(loc=mu, scale=sigma)

        return dist, mu, sigma


class LatentModel(nn.Module):
    def __init__(self, num_hidden, d_x=1, d_y=1):
        super(LatentModel, self).__init__()
        self.latent_encoder = LatentEncoder(num_hidden, num_hidden, d_x=d_x, d_y=d_y)
        self.deterministic_encoder = DeterministicEncoder(num_hidden, d_x=d_x, d_y=d_y)
        self.decoder = Decoder(num_hidden, d_x=d_x, d_y=d_y)
        self.BCELoss = nn.BCELoss()
        self.optim_state_dict = self.state_dict()

    @profile
    def forward(self, query, target_y=None):
        # LatentModel
        (context_x, context_y), target_x = query
        num_targets = target_x.size(1)

        n_context = context_x.shape[1]
        c, z = self.latent_encoder(context_x, context_y, target_x)
        r = self.deterministic_encoder(context_x, context_y, target_x)  # [B, T_target, H]

        # mu should be the prediction of target y
        dist, mu, sigma = self.decoder(r, z, c, target_x)

        # For Training
        if target_y is not None:
            log_p = dist.log_prob(target_y).sum(dim=-1)

            prior_g_mu, prior_g_sigma, prior_l_mu, prior_l_sigma = self.latent_encoder.latent_context(context_x, context_y)
            posterior_g_mu, posterior_g_sigma, posterior_l_mu, posterior_l_sigma = self.latent_encoder.latent_context(target_x, target_y)

            # global
            global_posterior = torch.distributions.Normal(loc=posterior_g_mu, scale=posterior_g_sigma)
            global_prior = torch.distributions.Normal(loc=prior_g_mu, scale=prior_g_sigma)
            global_kl = torch.distributions.kl_divergence(global_posterior, global_prior).sum(dim=-1, keepdims=True)
            global_kl = global_kl.repeat([1, num_targets])


            # local
            posterior_l_mu = posterior_l_mu.mean(dim=1, keepdims=True).repeat([1, n_context, 1])
            posterior_l_sigma = posterior_l_sigma.mean(dim=1, keepdims=True).repeat([1, n_context, 1])
            local_posterior = torch.distributions.Normal(loc=posterior_l_mu, scale=posterior_l_sigma)

            local_prior = torch.distributions.Normal(loc=prior_l_mu, scale=prior_l_sigma)

            local_kl = torch.distributions.kl_divergence(local_posterior, local_prior).sum(dim=[1, 2])
            local_kl = local_kl.unsqueeze(1).repeat([1, num_targets])

            loss = - (log_p - global_kl / torch.tensor(num_targets).float() - local_kl / torch.tensor(num_targets).float()).mean()

        # For Generation
        else:
            log_p = None
            global_kl = None
            local_kl = None
            loss = None

        return mu, sigma, log_p, global_kl, local_kl, loss

    def save_to_optim(self):
        self.optim_state_dict = self.state_dict()

    def load_from_optim(self):
        self.load_state_dict(self.optim_state_dict)




class MyModel(Module):
    def __init__(self, in_dim, out_dim, configs,):
        super(MyModel, self).__init__()
        c = configs
        self.cost_rate = c.cost_rate
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout_r = c.dropout_r
        self.random_guide_weight = c.random_guide_weight

        if c.base_weight is not None:
            self.guide_weight = torch.FloatTensor([c.base_weight])
        else:
            self.guide_weight = torch.ones(1, out_dim, dtype=torch.float32) / out_dim
        self.hidden_layers = nn.ModuleList()

        h_in = in_dim
        for h_out in c.hidden_dim:
            self.hidden_layers.append(XLinear(h_in, h_out))
            h_in = h_out

        self.out_layer = XLinear(h_out, out_dim)

        # asset allocation
        self.aa_hidden_layer = XLinear(out_dim * 3 + in_dim, (out_dim * 3 + in_dim)//2)
        self.aa_hidden_layer2 = XLinear((out_dim * 3 + in_dim) // 2, out_dim)
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
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout_r, training=mask)

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


    def run(self, features, wgt, n_samples):
        x = torch.cat([features['idx'], features['macro']], dim=-1)
        pred = self.sample_predict(x, n_samples=n_samples)
        pred_mu = torch.mean(pred, dim=0)
        pred_sigma = torch.std(pred, dim=0)

        x = torch.cat([pred_mu, pred_sigma, wgt, x], dim=-1)
        x = self.aa_hidden_layer(x)
        x = F.relu(x)
        x = self.aa_hidden_layer2(x)
        x = F.relu(x)
        x = self.aa_out_layer(x)
        # x = F.elu(x + features['wgt'], 0.01) + 0.01 + min_wgt
        # x = F.softmax(x)
        x = F.sigmoid(x) + 0.001
        x = x / x.sum(dim=1, keepdim=True)

        return x, pred_mu, pred_sigma

    @profile
    def forward_with_loss(self, features, labels=None, n_samples=1000, loss_wgt=None, features_prev=None, is_train=True):
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

        if features_prev is not None:
            with torch.set_grad_enabled(False):
                prev_x, _, _ = self.run(features_prev, features['wgt'], n_samples)
        else:
            prev_x = features['wgt']

        x, pred_mu, pred_sigma = self.run(features, prev_x, n_samples)

        if torch.isnan(x).sum() > 0:
            for n, p in list(self.named_parameters()):
                print(n, '\nval:\n', p, '\ngrad:\n', p.grad)

            return False

        # guide_wgt = self.guide_weight.repeat(len(x), 1)
        # if is_train and self.random_guide_weight > 0:
        #     replace_idx = np.random.choice(np.arange(len(x)), int(len(x) * self.random_guide_weight), replace=False)
        #     guide_wgt[replace_idx] = torch.rand(len(replace_idx), guide_wgt.shape[1])
        # guide_wgt = guide_wgt.to(x.device)

        if is_train:
            if np.random.rand() > self.random_guide_weight:
                guide_wgt = self.guide_weight.repeat(len(x), 1).to(x.device)
            else:
                guide_wgt = torch.rand_like(x).to(x.device)
                guide_wgt = guide_wgt / guide_wgt.sum(dim=1, keepdim=True)
        else:
            guide_wgt = self.guide_weight.repeat(len(x), 1).to(x.device)

        if labels is not None:
            next_y = torch.exp(1 + labels['logy']) - 1.
            losses_dict = dict()
            # losses_dict['y_pf'] = -(x * labels['logy']).sum()
            losses_dict['y_pf'] = -((x - features['wgt']) * next_y).sum()
            losses_dict['mdd_pf'] = F.elu(-(x * next_y).sum(dim=1) - 0.05, 1e-6).sum()
            # losses_dict['mdd_pf'] = torch.relu(-(x * next_y).sum(dim=1) - 0.05).sum()
            losses_dict['logy'] = self.loss_func_logy(pred_mu, labels['logy'])
            losses_dict['wgt'] = nn.KLDivLoss(reduction='sum')(torch.log(x), labels['wgt'])
            losses_dict['wgt2'] = nn.KLDivLoss(reduction='sum')(torch.log(x), features['wgt'])
            losses_dict['wgt_guide'] = nn.KLDivLoss(reduction='sum')(torch.log(x), guide_wgt)
            losses_dict['cost'] = torch.abs(x - prev_x).sum() * self.cost_rate
            # losses_dict['cost'] = torch.abs(x - features['wgt']).sum() * self.cost_rate

            if loss_wgt is not None:
                losses_dict['entropy'] = -HLoss()(x)
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



