from abc import ABCMeta, abstractmethod
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AbstractDataset(metaclass=ABCMeta):
    base_dir = './data/'

    @classmethod
    def set_base_dir(cls, dir):
        cls.base_dir = dir

    def get_base_dir(self):
        return self.base_dir

    def get_file_dir(self, file_nm):
        return os.path.join(self.base_dir, file_nm)

    @abstractmethod
    def read_dataset_from_file(self):
        pass


class MyDataset(AbstractDataset):
    def __init__(self, dataset=None, file_dir=[]):
        self.file_dir = file_dir
        self.dataset = dataset

    def read_dataset_from_file(self, file_nm, use_lower_columns=True):
        predefined_sep = dict(txt='\t', csv=',')

        file_dir = self.get_file_dir(file_nm)
        sep = predefined_sep[file_dir.split('.')[-1]]
        dataset = pd.read_csv(file_dir, index_col=0, sep=sep)
        if use_lower_columns:
            dataset.columns = [col.lower() for col in dataset.columns]

        return dataset, file_dir

    def merge_dataset(self, other, merge_type='inner'):
        assert merge_type in ['inner', 'outer', 'left', 'right']
        merged = pd.merge(self.dataset, other.dataset, left_index=True, right_index=True, how=merge_type)
        return merged

    def __call__(self):
        return self.dataset

    def __add__(self, other):
        merged = self.merge_dataset(other, 'inner')
        merged_filedir = self.file_dir + other.file_dir
        return MyDataset(merged)


class MacroDataset(MyDataset):
    def __init__(self, file_nm='macro_data_20200615.txt'):

        super().__init__(dataset=dataset)


    def transform(self):
        pass


class AplusDataset(MyDataset):
    def __init__(self, file_nm='app_data_20200615.txt'):
        self.file_nm = file_nm
        dataset = self.read_dataset_from_file(file_nm=file_nm)

        super().__init__(dataset=dataset)
