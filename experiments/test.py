

import torch
from torch import nn
import copy


class A(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)


x = torch.randn(1, 10)

a = A()
nn.init.constant_(a.layer.weight, 0.1)
b = copy.deepcopy(a.state_dict())
print(b)
nn.init.constant_(a.layer.weight, 0.01)
print(b)
