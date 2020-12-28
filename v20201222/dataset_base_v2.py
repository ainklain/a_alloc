
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset, RandomSampler, SequentialSampler

from v20201222 import transforms_v2


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
    def __init__(self, df=None, file_dir=[], label_columns_dict=OrderedDict()):
        self.df = self.idx = self.columns = None
        self.file_dir = file_dir
        self.label_columns_dict = label_columns_dict

        if df is not None:
            self.set_data(df)

    def set_data(self, df, use_lower_columns=True):
        if df is not None:
            self.df = df
            if use_lower_columns:
                self.df.columns = [col.lower() for col in df.columns]

            self.columns = list(self.df.columns)
            self.idx = list(self.df.index)

    def merge_data(self, other, merge_type='inner'):
        assert merge_type in ['inner', 'outer', 'left', 'right']
        merged = pd.merge(self.df, other.df,
                          left_index=True, right_index=True,
                          how=merge_type)
        return merged

    def _transform(self):
        pass

    def transform(self):
        with torch.set_grad_enabled(False):
            self.df = transforms_v2.ToNumpy()(self.df)
            self._transform()
            self.df = transforms_v2.ToDataFrame(self.idx, self.columns)(self.df)
            self.df.fillna(0)

    def __repr__(self):
        return_str = ""
        return_str += "[head]:\n{}\n[tail]:\n{}\n".format(self.df.head(2), self.df.tail(2))
        return_str += "file dir: {}\nshape: {}\n".format(self.file_dir, self.df.shape)
        return return_str

    def __len__(self):
        return len(self.df)

    def __call__(self):
        return self.df

    def __add__(self, other):
        if self.df is None:
            merged = other.df
        elif other.df is None:
            merged = self.df
        else:
            merged = self.merge_data(other, 'left')
            merged.ffill(inplace=True)

        merged_filedir = self.file_dir + other.file_dir

        merged_label_columns = OrderedDict(
            list(self.label_columns_dict.items()) + list(other.label_columns_dict.items()))

        return AddibleData(merged, merged_filedir, merged_label_columns)


class DataFromFiles(AddibleData):
    def __init__(self, file_nm='macro_data_20200615.txt'):
        super().__init__()
        self.set_data_from_name(file_nm)

    def get_file_dir(self, file_nm):
        return os.path.join(self.base_dir, file_nm)

    def set_data_from_name(self, file_nm, use_lower_columns=True):
        file_dir = self.get_file_dir(file_nm)
        self.set_data(self.read_data_from_file(file_dir), use_lower_columns)

        self.file_dir = [file_dir]

    @classmethod
    def read_data_from_file(cls, file_dir):
        predefined_sep = dict(txt='\t', csv=',')

        sep = predefined_sep[file_dir.split('.')[-1]]
        df = pd.read_csv(file_dir, index_col=0, sep=sep)

        return df


class DatasetForTimeSeriesBase(Dataset):
    def __init__(self, addible_data, sampling_days=20, k_days=20):
        self.sampling_days = sampling_days
        self.k_days = k_days

        self.set_data_info(addible_data)

    def set_data_info(self, addible_data):
        self.arr = np.array(addible_data.df, dtype=np.float32)
        self.idx = addible_data.idx
        self.columns = addible_data.columns
        self.label_columns_dict = addible_data.label_columns_dict

    @property
    def label_columns_idx(self):
        return [self.columns.index(idx) for lc_list in self.label_columns_dict.values() for idx in lc_list]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        out = {'features': self.arr[i, :],
               'labels': self.arr[i + self.k_days + 1, self.label_columns_idx]}

        return out


class DatasetForSingleTimestep(DatasetForTimeSeriesBase):
    def __init__(self, *args, **kwargs):
        super(DatasetForSingleTimestep, self).__init__(*args, **kwargs)
        self.default_range = [0, len(self.idx)]

    def __getitem__(self, i):
        out = {'features_prev': self.arr[i - self.sampling_days, :],
               'features': self.arr[i, :],
               'labels': self.arr[i + self.k_days + 1, self.label_columns_idx]}

        return out


class DatasetForMultiTimesteps(DatasetForTimeSeriesBase):
    def __init__(self, *args, window=250, **kwargs):
        super(DatasetForMultiTimesteps, self).__init__(*args, **kwargs)
        self.window = window
        self.adj = window % self.sampling_days
        self.default_range = [self.window + self.sampling_days, len(self.idx) - self.k_days]
        # self.default_range = [self.window + self.sampling_days, len(self.idx)]

    def __getitem__(self, i):
        if i >= self.default_range[1] or i < self.default_range[0]:
            label_arr = None
            # label_arr = self.arr[(i+1):(i + self.k_days + 2)][::self.sampling_days, self.label_columns_idx]
        else:
            label_arr = self.arr[(i+1):(i + self.k_days + 2)][::self.sampling_days, self.label_columns_idx]

        out = {'features_prev': self.arr[max(0, i - self.window - self.sampling_days):(i + 1 - self.sampling_days)][self.adj::self.sampling_days],
               'features': self.arr[max(0, i - self.window):(i + 1)][self.adj::self.sampling_days],
               'labels': label_arr,
               }

        return out


class DatasetManagerBase(metaclass=ABCMeta):
    def __init__(self, data_list, dataset_type='multi', **kwargs):
        # kwargs = dict(window=250, sampling_days=20, k_days=20)

        # left joined
        self.initialize(data_list, dataset_type, **kwargs)

    def initialize(self, data_list, dataset_type, **kwargs):
        self.transform(data_list)
        addible_data = self.merge(data_list)
        dataset_func = self.define_dataset_func(dataset_type)
        self.dataset = dataset_func(addible_data, **kwargs)

    def define_dataset_func(self, dataset_type):
        if dataset_type in ['multi', 'multitimesteps', 'multi_timesteps']:
            return DatasetForMultiTimesteps
        elif dataset_type in ['single', 'singletimestep', 'single_timestep']:
            return DatasetForSingleTimestep

    @property
    def max_len(self):
        return len(self.dataset)

    @property
    def features_list(self):
        return self.dataset.columns

    @property
    def labels_list(self):
        return self.dataset.columns

    def transform(self, dataset_list):
        for dataset in dataset_list:
            dataset.transform()

    def merge(self, dataset_list):
        # left join merged
        out = dataset_list[0]
        for dataset in dataset_list[1:]:
            out += dataset

        return out

    def get_sampler(self, sampler_type='random'):
        # print(sampler_type)
        sampler_type = sampler_type.lower()
        if sampler_type in ['random_sampler', 'random', 'randomsampler', 'random_with_replacement']:
            sampler_cls = RandomSampler
            sampler_kwargs = dict(replacement=True)
        elif sampler_type in ['random_without_replacement']:
            sampler_cls = RandomSampler
            sampler_kwargs = dict(replacement=False)
        elif sampler_type in ['sequential_sampler', 'sequential', 'sequentialsampler']:
            sampler_cls = SequentialSampler
            sampler_kwargs = dict()
        else:
            raise NotImplementedError

        return sampler_cls, sampler_kwargs

    def get_data_loader_default(self, **kwargs):
        return DataLoader(self.dataset, **kwargs)

    def get_data_loader(self, base_i, mode, **kwargs):
        params = self.mode_params(base_i, mode)

        # mode별 사용 index 정의 및 subset
        mode_range = np.arange(params['begin_i'], params['end_i'])
        sub_dataset = Subset(self.dataset, mode_range)

        # mode: train - random sampling with replacement / test - sequential sampling
        sampler_cls, sampler_kwargs = self.get_sampler(params['sampler_type'])
        sampler = sampler_cls(sub_dataset, **sampler_kwargs)

        if params['batch_size'] == -1:
            batch_size = len(sub_dataset)
        else:
            batch_size = params['batch_size']

        # shuffle must be False if sampler exists
        data_loader = DataLoader(sub_dataset, shuffle=False, sampler=sampler, batch_size=batch_size, **kwargs)

        return data_loader

    @abstractmethod
    def mode_params(self, base_i, mode):
        pass