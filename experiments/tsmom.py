
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

# from dataset_v2 import AplusDataWithoutTransform, MacroDataWithoutTransform
from dataset_v2 import AplusLogyData, MacroLogyData


def sequence_to_dataset(arr_p, np_split=False):
    n_timestep, n_asset = arr_p.shape
    if np_split:
        dataset_pos = []
        dataset_neg = []
        for i in range(n_asset):
            arr = arr_p[:, i]
            for t in range(250, n_timestep-21):
                label = int(arr[t] / arr[t-250] - 1 >= 0)
                next_y = arr[(t+20)] / arr[t+1] - 1
                if label == 1:
                    dataset_pos.append([arr[(t-250):(t+1)], label, next_y])
                else:
                    dataset_neg.append([arr[(t-250):(t+1)], label, next_y])
        return dataset_pos, dataset_neg
    else:
        dataset = []
        for i in range(n_asset):
            arr = arr_p[:, i]
            for t in range(250, n_timestep-21):
                label = int(arr[t] / arr[t-250] - 1 >= 0)
                next_y = arr[(t+20)] / arr[t+1] - 1
                dataset.append([arr[(t-250):(t+1)], label, next_y])

        return dataset


class MyDataset(Dataset):
    def __init__(self, arr_p, i=None, type_='train'):
        super().__init__()

        arr_p = arr_p.astype(np.float32)
        if type_ == 'train':
            arr_selected = arr_p[:, :10]
            dataset_pos, dataset_neg = sequence_to_dataset(arr_selected, np_split=True)
            arr_p_features = np.concatenate([d[0].reshape([1, -1]) for d in dataset_pos], axis=0)
            arr_p_nexty = np.array([d[2] for d in dataset_pos], dtype=np.float32)
            arr_n_features = np.concatenate([d[0].reshape([1, -1]) for d in dataset_neg], axis=0)
            arr_n_nexty = np.array([d[2] for d in dataset_neg], dtype=np.float32)

            # downsamples
            # selected_idx = list(np.random.choice(len(dataset_pos), len(dataset_neg), replace=False))
            # upsamples
            selected_idx = list(np.random.choice(len(dataset_neg), len(dataset_pos), replace=True))

            arr_features = np.concatenate([arr_p_features, arr_n_features[selected_idx]], axis=0)
            arr_labels = np.concatenate([np.ones([len(selected_idx)], dtype=np.int64),
                                         np.zeros([len(selected_idx)], dtype=np.int64)], axis=0)
            arr_nexty = np.concatenate([arr_p_nexty, arr_n_nexty[selected_idx]], axis=0)

            # shuffled_idx = np.random.choice(len(arr_features), len(arr_features), replace=False)
        else:
            arr_selected = arr_p[:, i:(i+1)]
            dataset = sequence_to_dataset(arr_selected, np_split=False)
            arr_features = np.concatenate([d[0].reshape([1, -1]) for d in dataset], axis=0)[::20]
            arr_labels = np.array([d[1] for d in dataset], dtype=np.int64)[::20]
            arr_nexty = np.array([d[2] for d in dataset], dtype=np.float32)[::20]

        arr_features = arr_features - np.mean(arr_features, axis=1, keepdims=True)
        self.arr_features, self.arr_labels, self.arr_nexty = arr_features, arr_labels, arr_nexty

    def __len__(self):
        return len(self.arr_features)

    def __getitem__(self, i):
        return self.arr_features[i], self.arr_labels[i], self.arr_nexty[i]


def get_rawdata():
    logy_data = [AplusLogyData(), MacroLogyData()]
    for data_type in logy_data:
        data_type.transform()

    df = pd.merge(*[x.df for x in logy_data], left_index=True, right_index=True, how='inner')

    df = df[~df.isna().any(axis=1)]
    return df



class MaskedConv1d(nn.Module):
    def __init__(self, *args, drop_rate=0.5, **kwargs):
        super().__init__()
        self.drop_rate = drop_rate
        self.conv1d = nn.Conv1d(*args, **kwargs)

    def forward(self, x):
        if self.training:
            masked = torch.rand_like(self.conv1d.weight) < self.drop_rate
            weight = torch.masked_fill(self.conv1d.weight, masked, 0.)
        else:
            weight = self.conv1d.weight

        return self.conv1d(x)
        # return F.conv1d(x, weight, stride=self.conv1d.stride, padding=self.conv1d.padding)


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        dim_in = 1
        dim_kernels = [5, 4, 3, 4]
        n_strides = [5, 1, 1, 1]
        dim_latent = 64

        modules = []

        # for i in range(len(dim_kernels)):
        #     modules.append(nn.Sequential(
        #         # nn.ConstantPad1d([dim_k-1, 0], 0.),
        #         MaskedConv1d(dim_in, dim_latent, drop_rate=0.0, kernel_size=dim_kernels[i], stride=n_strides[i]),
        #         nn.BatchNorm1d(dim_latent),
        #         nn.LeakyReLU()
        #     ))
        #     dim_in = dim_latent
        #
        # self.encoder = nn.Sequential(*modules)

        self.classifier = nn.Sequential(
            nn.Linear(251, 64),
            # nn.Linear(64 * 42, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2)
            # nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.unsqueeze(1)
        # x = self.encoder(x)
        # x = x.flatten(start_dim=1)
        return self.classifier(x)

    def loss_func(self, y_pred, y):
        y_pred = torch.log_softmax(y_pred, -1)
        loss_func = nn.NLLLoss()
        # loss_func = nn.BCELoss()
        return loss_func(y_pred, y)





def np_ify(x):
    return x.detach().cpu().numpy()


def train(ep, train_loader, model, optimizer):
    cnt = 0
    accs = 0
    losses = 0
    for feature, label, next_y in train_loader:
        # feature, label, next_y = next(iter(train_loader))
        # feature = feature.unsqueeze(1).float().cuda()
        feature = feature.float().cuda()
        label = label.long().cuda()
        optimizer.zero_grad()
        y_pred = model(feature)
        loss = model.loss_func(y_pred, label)
        loss.backward()
        optimizer.step()
        losses += np_ify(loss)
        accs += np_ify(torch.eq(y_pred.argmax(dim=1), label).sum())
        cnt += len(label)
        # print(np_ify(loss), sum(np_ify(torch.eq(y_pred.argmax(dim=1), label))) / len(label))

    losses /= len(train_loader)
    accs /= cnt

    print("[ep: {}] loss: {:.6f} acc: {:.4f}".format(ep, losses, accs))


def test(test_loader, model):
    y_pf = []
    yy = []
    y_pp = []
    ans = []
    model.to('cpu')

    cnt = 0
    accs = 0
    with torch.set_grad_enabled(False):
        for feature, label, next_y in test_loader:
            feature, label, next_y = feature[0:1], label[0:1], next_y[0:1]
            # feature, label, next_y = next(iter(test_loader))
            feature = feature.float()
            label = label.long()
            y_pred = model(feature)
            ans.append(y_pred)
            yy.append(np_ify(next_y)[0])
            y_pp.append(np_ify(next_y * label)[0])
            y_pf.append(np_ify(next_y * (y_pred.argmax() == 1))[0])

            accs += np_ify(torch.eq(y_pred.argmax(dim=1), label).sum())
            cnt += len(label)

    accs /= cnt
    print("acc: {:.4f}".format(accs))

    yy_cum = (np.array(yy) + 1).cumprod()
    y_pp_cum = (np.array(y_pp) + 1).cumprod()
    y_pf_cum = (np.array(y_pf) + 1).cumprod()

    df = pd.DataFrame({'y_base': yy_cum, 'y_ts': y_pp_cum, 'y_pf': y_pf_cum})

    plt.plot(df)
    plt.legend(df.columns)


def main():
    df = get_rawdata()
    df_p = (1+df).cumprod() / (1+df.iloc[0])
    arr_y = df.values
    arr_p = df_p.values

    train_dataset = MyDataset(arr_p, type_='train')
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


    # train_loader = toy_example()

    model = MyModel()
    model.to('cuda:0')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(100):
        train(ep, train_loader, model, optimizer)

    for i in range(arr_p.shape[1]):
        test_dataset = MyDataset(arr_p, i=i, type_='test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test(test_loader, model)
