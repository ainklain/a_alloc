
import numpy as np
import torch
from torch import nn, distributions
from torch.nn import init, Module, functional as F
from torch.distributions import Normal

# # #### profiler start ####
import builtins

try:
    builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func):
        return func

    builtins.profile = profile


def toy_data():
    # x = np.linspace(-3, 19, 500).astype(dtype=np.float32)
    x = np.random.rand(2048) * (19 + 3) - 3
    y = 2 * (np.sin(x/17 * 2 * np.pi)) * (x >= 0) + 0.1 * x * (x < 0) + 2.2 + np.random.randn(len(x)) * 0.01
    x = x.astype(dtype=np.float32).reshape([-1, 1])
    y = y.astype(dtype=np.float32).reshape([-1, 1])
    return x, y


from torch.utils.data import Dataset, DataLoader


class ToyDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def get_data_loader(mode='train'):
    x, y = toy_data()
    if mode == 'train':
        dataset = ToyDataset(x[:int(len(x) * 0.9)], y[:int(len(x)* 0.9)])
        loader = DataLoader(dataset, batch_size=128, shuffle=True)
    else:
        dataset = ToyDataset(x[:], y[:])
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return loader


def get_data_loader2(mode='train'):
    x, y = toy_data()
    train_dataset = ToyDataset(x[:int(len(x) * 0.6)], y[:int(len(x) * 0.6)])
    # train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, pin_memory=True, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, pin_memory=True, num_workers=0)

    test_dataset = ToyDataset(x[:], y[:])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_loader, test_loader


@profile
def test():
    import matplotlib.pyplot as plt


    # train_loader = get_data_loader('train')
    # test_loader = get_data_loader('test')

    train_loader, test_loader = get_data_loader2()


    model = ExpectedReturnEstimatorSingleShot(1, 1, [16, 16])
    model.to('cuda:0')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    loss_func = nn.MSELoss(reduction='sum')

    for i in range(1001):

        if i % 1000 == 0:
            model.eval()
            after = []
            for x, y in test_loader:
                x, y = x.to('cuda:0'), y.to('cuda:0')
                with torch.no_grad():
                    out = model(x, False)
                    mu, sig = model.predict(x)
                after.append([x.detach().cpu().numpy()[0][0], y.detach().cpu().numpy()[0][0], out.cpu().numpy()[0][0], mu.cpu().numpy()[0][0], sig.cpu().numpy()[0][0]])

            after = np.array(after)
            after = after[after[:, 0].argsort()]

            fig = plt.figure()
            plt.plot(after[:, 0], after[:, 1], '.')
            plt.plot(after[:, 0], after[:, 2], '*')
            plt.plot(after[:, 0], after[:, 3])
            plt.plot(after[:, 0], after[:, 3] + 2 * after[:, 4], '-')
            plt.plot(after[:, 0], after[:, 3] - 2 * after[:, 4], '-')
            fig.savefig('./out/abc1_{}.jpg'.format(i))
            plt.close(fig)

        model.train()
        losses = 0
        for x, y in train_loader:
            x, y = x.to('cuda:0'), y.to('cuda:0')
            out = model(x)
            loss = loss_func(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                losses += loss.detach().cpu().numpy()

        if i % 100 == 0:
            print(i, losses)

@profile
def speed():
    a = torch.randn(100, 1)

    for _ in range(100):
        a.square()
        a * a
        a ** 2
        torch.pow(a, 2)


# # #### profiler end ####
class ExpectedReturnEstimatorSingleShot(Module):
    def __init__(self, in_dim, out_dim, hidden_dim, p_drop=0.3):
        super(ExpectedReturnEstimatorSingleShot, self).__init__()

        self.p_drop = p_drop
        self.hidden_layers = nn.ModuleList()

        h_in = in_dim
        for h_out in hidden_dim:
            self.hidden_layers.append(nn.Linear(h_in, h_out))
            h_in = h_out

        self.out_layer = nn.Linear(h_out, out_dim)

    def forward(self, x, sample=True):
        """
        x = torch.randn(200, 512, 30).cuda()
        """
        for h_layer in self.hidden_layers:
            x = h_layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.p_drop, training=self.training) * (1 - self.p_drop)

        x = self.out_layer(x)
        return x

    @profile
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
                sqrt_pred_var = torch.sqrt(pred_var)
                mu_std_ratio = pred_mu / sqrt_pred_var
                cdf_value = normal_dist.cdf(mu_std_ratio)
                pdf_value = torch.exp(normal_dist.log_prob(mu_std_ratio))
                pred_mu_new = pred_mu * cdf_value + sqrt_pred_var * pdf_value
                pred_var_new = (pred_mu * pred_mu + pred_var) * cdf_value + pred_mu * pred_var * pred_var * pdf_value - pred_mu_new * pred_mu_new
                pred_mu, pred_var = pred_mu_new, pred_var_new

                # dropout
                pred_mu = pred_mu * (1 - p_drop)
                pred_var = pred_var * p_drop * (1 - p_drop) + pred_var * (1 - p_drop) ** 2 + pred_mu * pred_mu * p_drop * (1 - p_drop)

            pred_mu = self.out_layer(pred_mu)
            pred_sigma = torch.sqrt(torch.mm(pred_var, self.out_layer.weight.data.square().t()))
        return pred_mu, pred_sigma

    def sample(self, x):
        pred_mu, pred_sigma = self.predict(x)
        z = torch.randn_like(pred_mu)
        return pred_sigma * z + pred_mu


if __name__ == '__main__':
    test()
    # speed()