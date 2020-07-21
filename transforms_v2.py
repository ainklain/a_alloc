
from abc import ABCMeta, abstractmethod
from functools import partial
import pandas as pd
import numpy as np


class BaseTransform:
    @abstractmethod
    def forward(self, x):
        """
            x : [n_timesteps, n_features]
        """
        pass

    def __call__(self, x):
        return self.forward(x)


class ToNumpy(BaseTransform):
    def forward(self, x):
        return np.array(x)


class ToDataFrame(BaseTransform):
    def __init__(self, idx, columns):
        self.idx = idx
        self.columns = columns

    def forward(self, x):
        return pd.DataFrame(x, index=self.idx, columns=self.columns)


class Rolling(BaseTransform):
    def __init__(self, window):
        self.window = window

    def rolling(self, func, x):
        n_timesteps, n_features = x.shape
        y = np.zeros_like(x)
        for i in range(n_timesteps):
            y[i, :] = func(x[max(0, i-self.window):(i+1)])

        return y


class RollingMean(Rolling):
    def __init__(self, window=500):
        super(RollingMean, self).__init__(window)

    def forward(self, x):
        return self.rolling(np.nanmean, x)


class RollingStd(Rolling):
    def __init__(self, window=500):
        super(RollingStd, self).__init__(window)

    def forward(self, x):
        return self.rolling(partial(np.nanstd, ddof=1), x)


class RollingSharpe(Rolling):
    def __init__(self, window=500):
        super(RollingSharpe, self).__init__(window)

    def forward(self, x):
        func = lambda x: np.nanmean(x) / np.nanstd(x, ddof=1)
        return self.rolling(func, x)


class RollingNormalize(Rolling):
    def __init__(self, window=500):
        super(RollingNormalize, self).__init__(window)

    def forward(self, x):
        func = lambda x: ((x - np.nanmean(x)) / np.nanstd(x, ddof=1))[-1, :]
        return self.rolling(func, x)


class RollingReturn(Rolling):
    def __init__(self, window=20):
        super(RollingReturn, self).__init__(window)

    def forward(self, x):
        func = lambda x: x[-1, :] / x[0, :] - 1.
        return self.rolling(func, x)


class RollingLogReturn(Rolling):
    def __init__(self, window=20):
        super(RollingLogReturn, self).__init__(window)

    def forward(self, x):
        func = lambda x: np.log(x[-1, :] / x[0, :])
        return self.rolling(func, x)


class Transforms:
    def __init__(self, transforms_list=[]):
        self.transforms_list = transforms_list

    def sequential(self, x):
        for transforms_func in self.transforms_list:
            x = transforms_func(x)

        return x

    def apply(self, x, columns, suffix=None, reduce='none'):
        assert reduce in ['none', 'concat']
        if suffix:
            assert len(self.transforms_list) == len(suffix)
            iter_ = zip(suffix, self.transforms_list)
        else:
            iter_ = enumerate(self.transforms_list)

        y = []
        new_columns = []
        for i, transforms_func in iter_:
            y.append(transforms_func(x))
            new_columns.append(["{}_{}".format(c, str(i)) for c in columns])

        if reduce == 'concat':
            y = np.concatenate(y, axis=-1)
            new_columns = sum(new_columns, [])

        return y, new_columns


