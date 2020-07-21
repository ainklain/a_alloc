
from abc import ABCMeta, abstractmethod
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import transforms_v2


class AbstractData(metaclass=ABCMeta):
    base_dir = './data/'

    @classmethod
    def set_base_dir(cls, dir):
        cls.base_dir = dir

    def get_base_dir(self):
        return self.base_dir

    @abstractmethod
    def _transform(self):
        pass


class AddibleData(AbstractData):
    def __init__(self, df=None, file_dir=[]):
        self.df = self.idx = self.columns = None
        self.file_dir = file_dir

        if df is not None:
            self.set_data(df)

    def set_data(self, df):
        if df is not None:
            self.df = df
            self.idx = list(self.df.index)
            self.columns = list(self.df.columns)

    def merge_data(self, other, merge_type='inner'):
        assert merge_type in ['inner', 'outer', 'left', 'right']
        merged = pd.merge(self.df, other.df,
                          left_index=True, right_index=True,
                          how=merge_type)
        return merged

    def _transform(self):
        pass

    def transform(self):
        self.df = transforms_v2.ToNumpy()(self.df)
        self._transform()
        self.df = transforms_v2.ToDataFrame(self.idx, self.columns)(self.df)

    def __repr__(self):
        return_str = ""
        return_str += "file dir: {}\nshape: {}\n".format(self.file_dir, self.df.shape)
        return_str += "[head]:\n{}\n[tail]:\n{}\n".format(self.df.head(5), self.df.tail(5))
        return return_str

    def __len__(self):
        return len(self.df)

    def __call__(self):
        return self.df

    def __add__(self, other):
        if not self.file_dir:
            return AddibleData(other.df, other.file_dir)
        elif not other.file_dir:
            return AddibleData(self.df, self.file_dir)
        else:
            merged = self.merge_data(other, 'left')
            merged.ffill(inplace=True)
            merged_filedir = self.file_dir + other.file_dir

            return AddibleData(merged, merged_filedir)


class DataFromFiles(AddibleData):
    def __init__(self, file_nm='macro_data_20200615.txt'):
        super().__init__()
        self.set_data_from_name(file_nm)

    def get_file_dir(self, file_nm):
        return os.path.join(self.base_dir, file_nm)

    def set_data_from_name(self, file_nm):
        file_dir = self.get_file_dir(file_nm)
        self.df = self.read_data_from_file(file_dir)
        self.idx = list(self.df.index)
        self.columns = list(self.df.columns)

        self.file_dir = [file_dir]

    @classmethod
    def read_data_from_file(cls, file_dir, use_lower_columns=True):
        predefined_sep = dict(txt='\t', csv=',')

        sep = predefined_sep[file_dir.split('.')[-1]]
        df = pd.read_csv(file_dir, index_col=0, sep=sep)
        if use_lower_columns:
            df.columns = [col.lower() for col in df.columns]

        return df


class DatasetForSingleTimesteps(Dataset):
    def __init__(self, df, sampling_days=20, k_days=20):
        self.idx = list(df.index)
        self.columns = list(df.columns)
        self.arr = np.array(df)
        self.sampling_days = sampling_days
        self.k_days = k_days
        self.default_range = [0, len(self.idx)]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        out = {'features': self.arr[i, :],
               'labels': self.arr[i+self.k_days+1, :]}

        return out


class DatasetForMultiTimesteps(Dataset):
    def __init__(self, df, window=250, sampling_days=20, k_days=20):
        self.idx = list(df.index)
        self.columns = list(df.columns)
        self.arr = np.array(df)
        self.window = window
        self.sampling_days = sampling_days
        self.k_days = k_days
        self.adj = window % sampling_days
        self.default_range = [self.window, len(self.idx) - self.k_days]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        out = {'features':  self.arr[max(0, i - self.window):(i + 1)][self.adj::self.sampling_days],
               'labels':  self.arr[(i+1):(i + self.k_days + 2)][::self.sampling_days]}

        return out


class DatasetManagerBase:
    def __init__(self, data_list, dataset_type='multi_step', **kwargs):
        # kwargs = dict(window=250, sampling_days=20, k_days=20)

        # left joined
        self.initialize(data_list, dataset_type, **kwargs)

    def initialize(self, data_list, dataset_type, **kwargs):
        self.transform(data_list)
        df = self.merge(data_list).df
        if dataset_type in ['single_step', 'singlestep', 'single']:
            self.dataset = DatasetForSingleTimesteps(df, **kwargs)

        elif dataset_type in ['multi_step', 'multistep', 'multisteps', 'multi']:
            self.dataset = DatasetForMultiTimesteps(df, **kwargs)

    def transform(self, dataset_list):
        for dataset in dataset_list:
            dataset.transform()

    def merge(self, dataset_list):
        # left join merged
        out = dataset_list[0]
        for dataset in dataset_list[1:]:
            out += dataset

        return out

    def get_data_loader(self, **kwargs):
        return DataLoader(self.dataset, **kwargs)

