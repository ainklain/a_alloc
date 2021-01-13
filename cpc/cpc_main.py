
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# from dataset_v2 import AplusDataWithoutTransform, MacroDataWithoutTransform
from dataset_v2 import AplusLogyData, MacroLogyData


def np_ify(x):
    return x.detach().cpu().numpy()


class SampleDataset(Dataset):
    def __init__(self, df_raw):
        super().__init__()
        self.asset_name = list(df_raw.columns)
        self.arr = df_raw.to_numpy(dtype=np.float32).transpose()[:, np.newaxis, :]

    def name_to_idx(self, name):
        return self.asset_name.index(name)

    def num_timesteps(self):
        return self.arr.shape[-1]

    def __len__(self):
        return len(self.asset_name)

    def __getitem__(self, idx):
        return self.arr[idx]


class SampleDataset_new(Dataset):
    def __init__(self, df_raw, type_='train'):
        super().__init__()
        self.num_timesteps_data = 1000
        self.num_timesteps_label = 400
        self.asset_name = list(df_raw.columns)
        self.arr = df_raw.to_numpy(dtype=np.float32).transpose()[:, np.newaxis, :]
        # arr_p = np.cumprod((1+arr), axis=-1)
        # arr_p = np.concatenate([arr_p[:, :, 20:] / arr_p[:, :, :-20], arr_p[:, :, -1:] / arr_p[:, :, -20:]], axis=-1)
        if type_ == 'train':
            self.idx_list = np.arange(self.num_timesteps_data, int(len(df_raw) * 0.6))
        elif type_ == 'eval':
            self.idx_list = np.arange(int(len(df_raw) * 0.6), int(len(df_raw * 0.8) - self.num_timesteps_label))
        elif type_ == 'test':
            self.idx_list = np.arange(int(len(df_raw) * 0.8), int(len(df_raw) - self.num_timesteps_label))
        else:
            raise NotImplementedError

    def name_to_idx(self, name):
        return self.asset_name.index(name)

    def num_timesteps(self):
        return self.arr.shape[-1]

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        base_t = self.idx_list[idx] - 1
        x = self.arr[:, :, (base_t - self.num_timesteps_data + 1):(base_t + self.num_timesteps_label + 1)]
        return x


def get_loader(dataset, batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_rawdata():
    # logy_data = [AplusLogyData(), MacroLogyData()]
    logy_data = [AplusLogyData()]
    for data_type in logy_data:
        data_type.transform()

    if len(logy_data) == 1:
        df = logy_data[0].df
    else:
        df = pd.merge(*[x.df for x in logy_data], left_index=True, right_index=True, how='inner')

    df = df[~df.isna().any(axis=1)]
    return df


class Encoder(nn.Module):
    def __init__(self, dim_hidden=64):
        super(Encoder, self).__init__()
        kernel_size = [5, 4, 3, 4]   # 5d, 20d, 60d, 240d

        self.encoder = nn.Sequential(
            nn.Conv1d(1, dim_hidden, kernel_size[0], stride=kernel_size[0], bias=False),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(inplace=True),
            nn.ConstantPad1d((kernel_size[1] - 1, 0), 0.),
            nn.Conv1d(dim_hidden, dim_hidden, kernel_size[1], stride=2, bias=False),
            nn.BatchNorm1d(
                dim_hidden),
            nn.LeakyReLU(inplace=True),
            nn.ConstantPad1d((kernel_size[2] - 1, 0), 0.),
            nn.Conv1d(dim_hidden, dim_hidden, kernel_size[2], stride=1, bias=False),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(inplace=True),
            nn.ConstantPad1d((kernel_size[3] - 1, 0), 0.),
            nn.Conv1d(dim_hidden, dim_hidden, kernel_size[3], stride=1, bias=False),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(inplace=True),
        )

    def get_factor(self, use_gpu=True):
        # input 대비 latent dim 비율
        x = torch.zeros(1, 1, 100)
        if use_gpu:
            x = x.cuda()

        y = self.encoder(x)
        return y.shape[-1] / 100

    def forward(self, x):
        """
        x: [n_assets, n_in_channel: 1, n_timesteps_x: 1000]
        out: [n_assets, n_hidden: 64, n_timesteps_z: 100] -> z ~ biweekly
        """
        x = F.pad(x, [-x.shape[-1] % 5, 0], 'constant', 0)  # 5의 배수가 되도록 왼쪽에 0으로 padding
        return self.encoder(x)


class CPCEncoder(nn.Module):
    def __init__(self, num_timesteps_data, num_timesteps_label, dim_hidden=64, dim_latent=32):
        super(CPCEncoder, self).__init__()
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        self.encoder = Encoder(dim_hidden)
        self.ar_model = nn.GRU(dim_hidden, dim_latent, num_layers=1, bidirectional=False, batch_first=True)

        self.enc_factor = self.encoder.get_factor(False)
        self.num_encoded_steps_data = int(num_timesteps_data * self.enc_factor)
        self.num_encoded_steps_label = int(num_timesteps_label * self.enc_factor)
        # self.linear_prediction = [nn.Linear(dim_latent, dim_hidden) for _ in range(self.num_encoded_steps_label)]
        self.linear_prediction = nn.Linear(dim_latent, dim_hidden * self.num_encoded_steps_label)

    def init_hidden(self, batch_size, use_gpu=True):
        h = torch.zeros(1, batch_size, self.dim_latent)
        if use_gpu:
            h = h.cuda()

        return h

    def forward(self, x, hidden):

        # x : [batch_size, 1, n_timesteps]
        nce = 0
        batch_size = len(x)

        # Encode
        z = self.encoder(x)     # z: [batch_size, dim_hidden, n_timesteps_z]
        z = z.transpose(1, 2)   # z: [batch_size, n_timesteps_z, dim_hidden]

        # z_futures
        z_futures = z[:, self.num_encoded_steps_data:, :].clone().detach().permute(1, 0, 2)  # z_futures: [n_timesteps_z, batch_size, dim_hidden]

        # Recurrent
        forward_seq = z[:, :self.num_encoded_steps_data, :]
        output, hidden = self.ar_model(forward_seq, hidden)    # z: [batch_size, n_timesteps_z, dim_latent]
        # c_t = output[:, (base_i-1):base_i, :]        # c_t: [batch_size, 1, dim_latent]
        c_t = output[:, -1, :]  # c_t: [batch_size, 1, dim_latent]

        pred = self.linear_prediction(c_t).reshape([len(c_t), -1, self.dim_hidden])
        total = torch.bmm(pred.permute(1, 0, 2), z_futures.permute(0, 2, 1))

        label = torch.arange(0, batch_size).reshape(1, -1).repeat(self.num_encoded_steps_label, 1).to(x)
        correct = torch.sum(torch.eq(torch.argmax(torch.softmax(total, dim=-1), dim=-1), label))
        nce += torch.sum(torch.diagonal(torch.log_softmax(total, dim=-1), dim1=-2, dim2=-1))

        nce /= -1. * batch_size * len(z_futures)
        accuracy = 1. * correct.item() / (batch_size * len(z_futures))

        return c_t, accuracy, nce, hidden

    def predict(self, x, hidden):
        # x : [n_batch, 1, n_timesteps]
        z = self.encoder(x)     # z: [n_batch, dim_hidden, n_timesteps_z]
        z = z.transpose(1, 2)   # z: [n_batch, n_timesteps_z, dim_hidden]
        z, hidden = self.ar_model(z, hidden)    # z: [n_batch, n_timesteps_z, dim_latent]

        return z[:, -1, :], hidden


class Decoder(nn.Module):
    def __init__(self, dim_hidden=64, dim_latent=32):

        super(Decoder, self).__init__()
        kernel_size = [5, 4, 4]

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(dim_latent, dim_hidden, kernel_size[0], bias=False),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose1d(dim_hidden, dim_hidden, kernel_size[1], stride=2, padding=1, bias=False),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose1d(dim_hidden, dim_hidden, kernel_size[2], stride=2, padding=1, bias=False),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(inplace=True))

    def pos_encoding(self, x):
        t = torch.linspace(0., 1., 20)
        t = t.expand(len(x), 1, -1)
        x = torch.cat((torch.sin(t), torch.cos(t), x), dim=1)
        return x

    def forward(self, context):
        # context: [batch_size, dim_latent]  == c_t in cpc_encoder
        c = context.unsqueeze(-1)     # context: [batch_size, dim_latent, 1]
        x_pred = self.decoder(c)        # x_pred: [batch_size, dim_latent, 20]
        x_pred = x_pred.mean(dim=1, keepdim=True)
        x_pred = self.pos_encoding(x_pred)
        return x_pred       # x_pred: [batch_size, 3, 20] :  2 positional encoding + 1 value


class Critic(nn.Module):
    def __init__(self, dim_hidden=64):
        super(Critic, self).__init__()
        kernel_size = [5, 3]   # 5d, 20d, 60d, 240d
        self.encoder = nn.Sequential(
            nn.Conv1d(3, dim_hidden, kernel_size[0], bias=False),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(dim_hidden, dim_hidden, kernel_size[1], stride=2, bias=False),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(inplace=True),
        )

        self.critic = nn.Sequential(
            nn.Linear(7 * dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z.flatten(start_dim=1)
        out = self.critic(z)
        return out


class MyModel_simple(nn.Module):
    def __init__(self, num_timesteps_data=1000, num_timesteps_label=400, dim_hidden=64, dim_latent=32):
        super(MyModel_simple, self).__init__()
        self.dim_hidden = dim_hidden

        self.num_timesteps_data = num_timesteps_data
        self.num_timesteps_label = num_timesteps_label
        self.cpc_encoder = CPCEncoder(num_timesteps_data, num_timesteps_label, dim_hidden, dim_latent)

        self.out_layer = nn.Sequential(nn.Linear(dim_latent, dim_latent),
                                       nn.ReLU(),
                                       nn.Linear(dim_latent, 1))

        self.finetune_loss = nn.BCELoss()

    def init_hidden(self, *args, **kwargs):
        return self.cpc_encoder.init_hidden(*args, **kwargs)

    def forward(self, x, hidden):
        context, hidden = self.cpc_encoder.predict(x, hidden)
        y = torch.sigmoid(self.out_layer(context)).squeeze()

        return context, y

    def loss_function(self, x):
        hidden = self.init_hidden(len(x), x.device)
        context, accuracy, nce, hidden = self.cpc_encoder(x, hidden)

        return context, accuracy, nce

    def finetune(self, x):
        true_ = x[:, :, -20:]
        label = (1 * ((1 + true_.squeeze()).log().sum(dim=-1) > 0)).float()  # 20d logy

        hidden = self.init_hidden(len(x), x.device)
        context, accuracy, _, hidden = self.cpc_encoder(x, hidden)
        pred = torch.sigmoid(self.out_layer(context)).squeeze()
        loss = self.finetune_loss(pred, label)

        acc = ((pred > 0.5) & (label == 1.)).sum() / len(label)
        return acc, loss, pred, label


class MyModel(nn.Module):
    def __init__(self, num_timesteps_data=1000, num_timesteps_label=400, dim_hidden=64, dim_latent=32):
        super(MyModel, self).__init__()
        self.dim_hidden = dim_hidden

        self.num_timesteps_data = num_timesteps_data
        self.num_timesteps_label = num_timesteps_label
        self.cpc_encoder = CPCEncoder(num_timesteps_data, num_timesteps_label, dim_hidden, dim_latent)
        self.decoder = Decoder(dim_hidden, dim_latent)
        self.critic = Critic(dim_hidden)

    def init_hidden(self, *args, **kwargs):
        return self.cpc_encoder.init_hidden(*args, **kwargs)

    def forward(self, x_raw, hidden, t=None):

        if t is None:
            t_samples = torch.randint(self.num_timesteps_data, x_raw.shape[-1] - self.num_timesteps_label, size=(1,)).long()
        else:
            t_samples = max(self.num_timesteps_data, t)

        x = x_raw[:, :, (t_samples - self.num_timesteps_data + 1):(t_samples + self.num_timesteps_label + 1)]
        context, accuracy, nce, hidden = self.cpc_encoder(x, hidden)
        decoded = self.decoder(context)

        return context, decoded

    def loss_function(self, x_raw):
        hidden = self.init_hidden(len(x_raw), x_raw.device)

        t_samples = torch.randint(self.num_timesteps_data, x_raw.shape[-1] - self.num_timesteps_label, size=(1,)).long()

        x = x_raw[:, :, (t_samples - self.num_timesteps_data + 1):(t_samples + self.num_timesteps_label + 1)]
        true_ = x_raw[:, :, (t_samples + 1):(t_samples + 20 + 1)]
        context, accuracy, nce, hidden = self.cpc_encoder(x, hidden)
        decoded = self.decoder(context)
        true_ = self.decoder.pos_encoding(true_)

        label = torch.cat([torch.ones(len(true_))])


class MyModel_cpc(nn.Module):
    def __init__(self, timestep, seq_len, dim_hidden=64):
        super(MyModel, self).__init__()
        # self.batch_size = batch_size
        self.timestep = timestep
        self.seq_len = seq_len
        self.dim_hidden = dim_hidden
        kernel_size = [5, 4, 3, 4]   # 5d, 20d, 60d, 240d
        n_padding = 5 - (seq_len % 5)

        # encode
        self.encoder = nn.Sequential(
            nn.ConstantPad1d((n_padding, 0), 0.),
            nn.Conv1d(1, dim_hidden, kernel_size[0], stride=kernel_size[0], bias=False),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(inplace=True),
            nn.ConstantPad1d((kernel_size[1]-1, 0), 0.),
            nn.Conv1d(dim_hidden, dim_hidden, kernel_size[1], bias=False),
            nn.BatchNorm1d(
                dim_hidden),
            nn.LeakyReLU(inplace=True),
            nn.ConstantPad1d((kernel_size[2]-1, 0), 0.),
            nn.Conv1d(dim_hidden, dim_hidden, kernel_size[2], bias=False),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(inplace=True),
            nn.ConstantPad1d((kernel_size[3]-1, 0), 0.),
            nn.Conv1d(dim_hidden, dim_hidden, kernel_size[3], bias=False),
            nn.BatchNorm1d(dim_hidden),
            nn.LeakyReLU(inplace=True),
        )
        self.gru = nn.GRU(dim_hidden, dim_hidden//2, num_layers=1, bidirectional=False, batch_first=True)
        self.Wk = nn.ModuleList([nn.Linear(dim_hidden//2, dim_hidden) for _ in range(timestep)])
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        kernel_size.reverse()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(dim_hidden, dim_hidden, kernel_size[0], bias=False),
            nn.LeakyReLU(),
            nn.ConstantPad1d((kernel_size[1]-1, 0), 0.),
            nn.ConvTranspose1d(dim_hidden, dim_hidden, kernel_size[1], padding=kernel_size[1]-1, bias=False),
            nn.LeakyReLU(),
            nn.ConstantPad1d((kernel_size[2]-1, 0), 0.),
            nn.ConvTranspose1d(dim_hidden, dim_hidden, kernel_size[2], padding=kernel_size[2]-1, bias=False),
            nn.LeakyReLU(),
            nn.ConstantPad1d((kernel_size[3]-1, 0), 0.),
            nn.ConvTranspose1d(dim_hidden, dim_hidden, kernel_size[3], stride=kernel_size[3], bias=False),
            nn.LeakyReLU(),
        )

    def init_hidden(self, batch_size, use_gpu=True):
        if use_gpu:
            return torch.zeros(1, batch_size, self.dim_hidden//2, requires_grad=True).cuda()
        else:
            return torch.zeros(1, batch_size, self.dim_hidden//2, requires_grad=True)

    def forward(self, x, hidden):
        """
        batch_size = 8
        seq_len = 2487
        timestep = 40
        self = MyModel(timestep, seq_len)
        """
        batch_size = x.shape[0]
        seq_len = x.shape[-1]
        down_freq = 5
        context_size = 1000
        base_t = context_size // down_freq

        nce = 0
        # t_samples = torch.randint(50, self.seq_len // 20 - self.timestep, size=(1,)).long()
        t_samples = torch.randint(context_size, seq_len - self.timestep * down_freq-1, size=(1,)).long()

        x_in = x[:, :, (t_samples-context_size+1):(t_samples+self.timestep * down_freq+1)]
        z = self.encoder(x_in)
        z = z.transpose(1, 2)

        encode_samples = torch.empty((self.timestep, batch_size, self.dim_hidden)).float()
        for i in range(self.timestep):
            encode_samples[i-1] = z[:, self.timestep + i, :].view(batch_size, self.dim_hidden)

        forward_seq = z[:, :base_t, :]
        # hidden = self.init_hidden(batch_size, False)
        output, hidden = self.gru(forward_seq, hidden)
        c_t = output[:, base_t-1, :]
        pred = torch.empty((self.timestep, batch_size, self.dim_hidden)).float()
        for i in range(self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)

        for i in range(self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, batch_size)))
            nce += torch.sum(torch.diag(self.log_softmax(total)))
        nce /= -1. * batch_size * self.timestep
        accuracy = 1. * correct.item() / batch_size

        return accuracy, nce, hidden

    def predict(self, x, hidden):
        z = self.encoder(x)
        z = z.transpose(1, 2)
        output, hidden = self.gru(z, hidden)

        return output, hidden


def train(epoch, model, optimizer, train_loader, use_gpu=True):
    log_interval = 1
    device = 'cuda:0' if use_gpu else 'cpu'

    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        hidden = model.init_hidden(len(data), use_gpu=use_gpu)
        acc, loss, hidden = model(data, hidden)

        loss.backward()
        optimizer.step()
        lr = optimizer.param_groups[0]['lr']
        # lr = optimizer.update_learning_rate()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%]\tlr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), lr, acc, loss.item()
            ))

    loss = loss.detach().cpu().numpy()
    return acc, loss


def validate(model,  data_loader, use_gpu=True):
    device = 'cuda:0' if use_gpu else 'cpu'

    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for data in data_loader:
            data = data.float().unsqueeze(1).to(device)
            hidden = model.init_hidden(len(data), use_gpu=use_gpu)
            acc, loss, hidden = model(data, hidden)

            total_loss += len(data) * loss
            total_acc += len(data) * acc

    total_loss /= len(data_loader.dataset)
    total_acc /= len(data_loader.dataset)

    print('==> Validationset: avg. loss: {:.4f}\tAccuracy: {:.4f}\n'.format(total_loss, total_acc))

    total_loss = total_loss.numpy()
    return total_acc, total_loss


def main():
    use_gpu = torch.cuda.is_available()
    device = 'cuda:0' if use_gpu else 'cpu'
    df_raw = get_rawdata()
    df_train = df_raw[:int(len(df_raw)*0.6)]
    df_test = df_raw[:]

    train_loader = get_loader(SampleDataset(df_train), batch_size=17)
    test_loader = get_loader(SampleDataset(df_test), batch_size=17)

    # x_raw = next(iter(train_loader)).float()
    # batch_size, _, seq_len = x.shape
    model = MyModel()
    # models = MyModel_cpc(timestep=40, seq_len=len(train_loader.dataset[0]), dim_hidden=64)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    losses = {'t': [], 'e': []}
    best_acc = 0
    best_loss = float('inf')
    best_epoch = -1
    for epoch in range(1, 30001):
        train_acc, train_loss = train(epoch, model, optimizer, train_loader, use_gpu)
        val_acc, val_loss = validate(model, test_loader, use_gpu)
        losses['t'].append(train_loss)
        losses['e'].append(val_loss)

        if val_acc > best_acc:
            best_acc = max(val_acc, best_acc)
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            best_epoch = epoch + 1


def train_simple(epoch, model, optimizer, data_loader, use_gpu=True, is_train=True):
    device = 'cuda:0' if use_gpu else 'cpu'

    model.train()

    acc_ep = 0.
    loss_ep = 0.
    with torch.set_grad_enabled(is_train):
        for batch_idx, data in enumerate(data_loader):
            data = data[0].float().to(device)

            context, acc, loss = model.loss_function(data)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            lr = optimizer.param_groups[0]['lr']
            # lr = optimizer.update_learning_rate()
            acc_ep += acc
            loss_ep += loss.item()

    acc_ep /= len(data_loader)
    loss_ep /= len(data_loader)
    if is_train:
        type_ = 'Train'
    else:
        type_ = 'Eval'

    print('{} Epoch: {}  lr:{:.5f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(type_, epoch, lr, acc_ep, loss_ep))

    return acc_ep, loss_ep


def main_simple():
    use_gpu = torch.cuda.is_available()
    device = 'cuda:0' if use_gpu else 'cpu'
    df_raw = get_rawdata()

    train_loader = get_loader(SampleDataset_new(df_raw, 'train'), batch_size=1)
    eval_loader = get_loader(SampleDataset_new(df_raw, 'eval'), batch_size=1)
    test_loader = get_loader(SampleDataset_new(df_raw, 'test'), batch_size=1, shuffle=False)

    # x_raw = next(iter(train_loader)).float()
    # batch_size, _, seq_len = x.shape
    model = MyModel_simple()
    # models = MyModel_cpc(timestep=40, seq_len=len(train_loader.dataset[0]), dim_hidden=64)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    finetune_optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = {'t': [], 'e': []}
    best_acc = 0
    best_loss = float('inf')
    best_epoch = -1

    # training
    for epoch in range(1, 3):
        train_acc, train_loss = train_simple(epoch, model, optimizer, train_loader, use_gpu, is_train=True)
        eval_acc, eval_loss = train_simple(epoch, model, optimizer, eval_loader, use_gpu, is_train=False)
        losses['t'].append(train_loss)
        losses['e'].append(eval_loss)

        if eval_acc > best_acc:
            best_acc = max(eval_acc, best_acc)
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            best_epoch = epoch + 1

    # finetuning
    for epoch in range(1, 2):

        with torch.set_grad_enabled(False):
            ft_eval_acc_ep, ft_eval_loss_ep = 0., 0.
            for x_data in eval_loader:
                x_data = x_data[0].to(device)
                ft_eval_acc, ft_eval_loss, pred, label = model.finetune(x_data)
                ft_eval_acc_ep += ft_eval_acc.item()
                ft_eval_loss_ep += ft_eval_loss.item()

        ft_eval_acc_ep /= len(eval_loader)
        ft_eval_loss_ep /= len(eval_loader)

        ft_train_acc_ep, ft_train_loss_ep = 0., 0.
        for x_data in train_loader:
            x_data = x_data[0].to(device)
            finetune_optimizer.zero_grad()
            ft_train_acc, ft_train_loss, pred, label = model.finetune(x_data)
            ft_train_loss.backward()
            finetune_optimizer.step()
            ft_train_acc_ep += ft_train_acc.item()
            ft_train_loss_ep += ft_train_loss.item()

        ft_train_acc_ep /= len(train_loader)
        ft_train_loss_ep /= len(train_loader)

        print("ep : {}   acc: {:.3f} / {:.3f}  loss: {:.3f} / {:.3f}".format(
            epoch, ft_train_acc, ft_eval_acc, ft_train_loss, ft_eval_loss))


    results = []
    labels = []
    model.eval()
    with torch.set_grad_enabled(False):
        for x_data in train_loader:
            # x_data = next(iter(test_loader))
            x_data = x_data[0].to(device)
            data, label = x_data[:, :, :1000], (1+x_data[:, :, 1001:1021]).log().sum(dim=-1).exp()

            hidden = model.init_hidden(len(data))
            _, pred = model(data, hidden)

            label = np_ify(label).reshape([-1, 1])
            pred = np_ify(pred).reshape([-1, 1])
            labels.append(label)
            results.append((label * (pred >= 0.5)))

    labels_arr = np.concatenate(labels, axis=1)
    results_arr = np.concatenate(results, axis=1)

    import matplotlib.pyplot as plt
    labels_cum = labels_arr[:, ::20].cumprod(axis=-1)
    results_cum = results_arr[:, ::20].cumprod(axis=-1)
    i = 0
    plt.plot(labels_cum[i])
    plt.plot(results_cum[i])








