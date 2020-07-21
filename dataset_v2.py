
import numpy as np
from torch.utils.data import RandomSampler, DataLoader, Subset

from dataset_base_v2 import DataFromFiles, DatasetManagerBase
import transforms_v2


class AplusData(DataFromFiles):
    def __init__(self, file_nm='app_data_20200615.txt'):
        super().__init__(file_nm)

    def _transform(self):
        transforms_apply = transforms_v2.Transforms([
            transforms_v2.RollingLogReturn(20),
            transforms_v2.RollingLogReturn(60),
            transforms_v2.RollingLogReturn(120),
            transforms_v2.RollingLogReturn(250),
        ])

        self.df, self.columns = transforms_apply.apply(
            self.df, self.columns,
            suffix=[20, 60, 120, 250],
            reduce='concat')


class MacroData(DataFromFiles):
    def __init__(self, file_nm='macro_data_20200615.txt'):
        super().__init__(file_nm)
        self.df['copper_gold_r'] = self.df['hg1 comdty'] / self.df['gc1 comdty']
        self.df['spx_dj'] = self.df['spx index'] / self.df['indu index']
        self.df['spx_rs'] = self.df['spx index'] / self.df['rty index']
        self.columns = self.df.columns

    def _transform(self):
        transforms = transforms_v2.Transforms([
            transforms_v2.RollingNormalize(500),
        ])

        self.df = transforms.sequential(self.df)


class DatasetManager(DatasetManagerBase):
    """
        data_list = [AplusData(), MacroData()]
        dataset_type = 'multi'
        test_days = 250
        batch_size = 32
        dm = DatasetManager(data_list, dataset_type, test_days, batch_size)
        dl_train = dm.get_data_loader(700, 'train')
        dl_test = dm.get_data_loader(700, 'test')
    """
    def __init__(self, data_list, dataset_type, test_days, batch_size, **kwargs):
        super(DatasetManager, self).__init__(data_list, dataset_type, **kwargs)
        self.test_days = test_days
        self.batch_size = batch_size

    def get_data_loader(self, base_i, mode, **kwargs):
        default_begin_i, default_end_i = self.dataset.default_range
        if mode == 'train':
            begin_i = 0
            end_i = int(base_i * 0.9)
            batch_size = self.batch_size
        elif mode == 'test':
            begin_i = base_i
            end_i = min(base_i + self.test_days, len(self.dataset))
            batch_size = 1

        sub_range = np.arange(max(default_begin_i, begin_i), min(default_end_i, end_i))
        sub_dataset = Subset(self.dataset, sub_range)
        if mode == 'train':
            sampler = RandomSampler(sub_dataset, replacement=True)
        else:
            sampler = None

        data_loader = DataLoader(sub_dataset, shuffle=False, sampler=sampler, batch_size=batch_size, **kwargs)

        return data_loader


"""
class MyDataset(Dataset):
    def __init__(self):
        self.df = np.array([[i * 10**j for j in range(3)] for i in range(100)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        return self.df[item]

a = MyDataset()
mask = [1 if i >= 10 and i<= 20 else 0 for i in range(100)]
sampler = WeightedRandomSampler(mask, 5, replacement=True)
data_loader = DataLoader(a, batch_size=5, sampler=sampler)

next(iter(data_loader))
"""