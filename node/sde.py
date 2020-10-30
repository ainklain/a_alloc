
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchsde

from torch import nn

batch_size, state_size, brownian_size = 32, 1, 2
t_size = 20

class SDE(nn.Module):
    noise_type = 'general'
    sde_type = 'ito'
    def __init__(self):
        super().__init__()
        self.mu = nn.Linear(state_size, state_size)
        self.sigma = nn.Linear(state_size, state_size * brownian_size)

    def f(self, t, y):
        return self.mu(y)

    def g(self, t, y):
        return self.sigma(y).view(batch_size, state_size, brownian_size)

sde = SDE()
y0 = torch.full((batch_size, state_size), 0.1)
ts = torch.linspace(0, 1, t_size)
ys = torchsde.sdeint(sde, y0, ts)
import torch
import torchsde

from torch import nn

batch_size, state_size, brownian_size = 32, 1, 2
t_size = 20

def to_np(x):
    return x.detach().cpu().numpy()

class SDE(nn.Module):
    noise_type = 'general'
    sde_type = 'ito'
    def __init__(self):
        super().__init__()
        self.mu = nn.Linear(state_size, state_size)
        self.sigma = nn.Linear(state_size, state_size * brownian_size)

    def f(self, t, y):
        return self.mu(y)

    def g(self, t, y):
        return self.sigma(y).view(batch_size, state_size, brownian_size)

sde = SDE()
loss_func = nn.MSELoss()
sde.cuda()
optimizer = torch.optim.Adam(sde.parameters(), lr=1e-3)

y0 = torch.full((batch_size, state_size), 0.1).cuda()
ts = torch.linspace(0, 1, t_size).cuda()
ys = torchsde.sdeint(sde, y0, ts)
ys = ys.squeeze()

y_true = 0.03 / 12 + np.random.normal(0, 1, 20) * 0.2 / np.sqrt(12)
y_true = torch.from_numpy(np.cumsum(y_true)).float()
y_true = y_true.unsqueeze(1).expand([-1, 32])

for i in range(100):
    optimizer.zero_grad()
    ys = torchsde.sdeint(sde, y0, ts)
    ys = ys.squeeze()
    loss = loss_func(ys, y_true.cuda())
    loss.backward()
    optimizer.step()
    print(i, to_np(loss))