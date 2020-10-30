import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# from dataset_v2 import AplusDataWithoutTransform, MacroDataWithoutTransform
from dataset_v2 import AplusLogyData, MacroLogyData


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


def get_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_rawdata():
    logy_data = [AplusLogyData(), MacroLogyData()]
    for data_type in logy_data:
        data_type.transform()

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
    def __init__(self, n_timesteps_to_predict_z, dim_hidden=64, dim_latent=32):

        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        self.encoder = Encoder(dim_hidden)
        self.ar_model = nn.GRU(dim_hidden, dim_latent, num_layers=1, bidirectional=False, batch_first=True)

        self.enc_factor = self.self.encoder.get_factor(False)
        n_timesteps = int(n_timesteps_to_predict_z * self.enc_factor)
        self.linear_prediction = [nn.Linear(dim_latent, dim_hidden) for _ in range(n_timesteps)]

    def init_hidden(self, batch_size, use_gpu=True):
        h = torch.zeros(1, batch_size, self.dim_latent)
        if use_gpu:
            h = h.cuda()

        return h

    def forward(self, x, hidden, base_t=1000):
        base_i = int(base_t * self.enc_factor)

        # x : [batch_size, 1, n_timesteps]
        nce = 0
        batch_size = len(x)

        # Encode
        z = self.encoder(x)     # z: [batch_size, dim_hidden, n_timesteps_z]
        z = z.transpose(1, 2)   # z: [batch_size, n_timesteps_z, dim_hidden]

        # z_futures
        z_futures = z[:, base_i:, :].clone().detach().permute(1, 0, 2)  # z_futures: [n_timesteps_z, batch_size, dim_hidden]

        # Recurrent
        forward_seq = z[:, :base_i, :]
        output, hidden = self.ar_model(forward_seq, hidden)    # z: [batch_size, n_timesteps_z, dim_latent]
        c_t = output[:, (base_i-1):base_i, :]        # c_t: [batch_size, 1, dim_latent]

        pred = torch.empty_like(z_futures)
        for i in range(len(z_futures)):
            pred[i] = self.linear_prediction[i](c_t)

            total = torch.mm(z_futures[i], torch.transpose(pred[i], 0, 1))
            correct = torch.sum(torch.eq(torch.argmax(torch.softmax(total, dim=-1), dim=0), torch.arange(0, batch_size)))
            nce += torch.sum(torch.diag(torch.log_softmax(total, dim=-1)))

        nce /= -1. * batch_size * len(z_futures)
        accuracy = 1. * correct.item() / batch_size

        return c_t, accuracy, nce, hidden

    def predict(self, x, hidden):
        # x : [n_batch, 1, n_timesteps]
        z = self.encoder(x)     # z: [n_batch, dim_hidden, n_timesteps_z]
        z = z.transpose(1, 2)   # z: [n_batch, n_timesteps_z, dim_hidden]
        z, hidden = self.ar_model(z, hidden)    # z: [n_batch, n_timesteps_z, dim_latent]

        return z, hidden


class Decoder(nn.Module):
    def __init__(self, dim_hidden=64, dim_latent=32):

        super(Encoder, self).__init__()
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

    def forward(self, context):
        # context: [batch_size, 1, dim_latent]  == c_t in cpc_encoder
        c = context.transpose(1, 2)     # context: [batch_size, dim_latent, 1]
        x_pred = self.decoder(c)        # x_pred: [batch_size, dim_latent, 20]
        x_pred = x_pred.mean(dim=1, keepdim=True)
        t = torch.linspace(0., 1., 20)
        t = t.expand(len(x_pred), 1, -1)
        x_pred = torch.cat((torch.sin(t), torch.cos(t), x_pred), dim=1)
        return x_pred       # x_pred: [batch_size, 3, 20] :  2 positional encoding + 1 value


class Critic(nn.Module):
    def __init__(self, dim_hidden=64):
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
        z = z.flatten()
        out = self.critic(z)
        return out


class MyModel(nn.Module):
    def __init__(self, n_timesteps_to_predict_z=400, dim_hidden=64, dim_latent=32):
        self.cpc_encoder = CPCEncoder(n_timesteps_to_predict_z, dim_hidden, dim_latent)
        self.decoder = Decoder(dim_hidden, dim_latent)
        self.critic = Critic(dim_hidden, dim_latent)

    def forward(self):
        pass


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
            return torch.zeros(1, batch_size, self.dim_hidden//2).cuda()
        else:
            return torch.zeros(1, batch_size, self.dim_hidden//2)

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
    df_test = df_raw[int(len(df_raw) * 0.6):]

    train_loader = get_loader(SampleDataset(df_train), batch_size=17)
    test_loader = get_loader(SampleDataset(df_test), batch_size=17)

    # x = next(iter(train_loader)).float().unsqueeze(1)`
    # batch_size, _, seq_len = x.shape

    model = MyModel(timestep=40, seq_len=len(train_loader.dataset[0]), dim_hidden=64)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    losses = {'t': [], 'e': []}
    best_acc = 0
    best_loss = float('inf')
    best_epoch = -1
    for epoch in range(1, 30001):
        train_acc, train_loss = train(epoch, model, optimizer, train_loader, use_gpu)
        val_acc, val_loss = validate(model,  test_loader, use_gpu)
        losses['t'].append(train_loss)
        losses['e'].append(val_loss)

        if val_acc > best_acc:
            best_acc = max(val_acc, best_acc)
            best_epoch = epoch + 1
        elif epoch - best_epoch > 2:
            best_epoch = epoch + 1









