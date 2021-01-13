
import numpy as np
from collections import OrderedDict
from dataset_base_v2 import DataFromFiles, DatasetManagerBase, DatasetForTimeSeriesBase
import transforms_v2


# Data Description


class AplusDataWithoutTransform(DataFromFiles):
    def __init__(self, file_nm='app_data_20201020.txt'):
        super().__init__(file_nm)


class MacroDataWithoutTransform(DataFromFiles):
    def __init__(self, file_nm='macro_data_20201020.txt'):
        super().__init__(file_nm)


class AplusLogyData(DataFromFiles):
    def __init__(self, file_nm='app_data_20201020.txt'):
        super().__init__(file_nm)

    def _transform(self):
        transforms_apply = transforms_v2.Transforms([
            (transforms_v2.RollingLogReturn(1), 'logy'),
        ])

        self.df = transforms_apply.sequential(self.df)



class MacroLogyData(DataFromFiles):
    def __init__(self, file_nm='macro_data_20201020.txt'):
        super().__init__(file_nm)

    def _transform(self):
        transforms_apply = transforms_v2.Transforms([
            (transforms_v2.RollingLogReturn(1), 'logy'),
        ])

        self.df = transforms_apply.sequential(self.df)


class AplusData(DataFromFiles):
    def __init__(self, file_nm='app_data_20200831.txt'):
        super().__init__(file_nm)
        self.label_columns_dict = OrderedDict()
        for feature in ['logy', 'mu', 'sigma']:
            self.label_columns_dict[feature] = ['{}_{}'.format(col, feature) for col in self.columns]

    def _transform(self):
        transforms_apply = transforms_v2.Transforms([
            # (transforms_v2.RollingLogReturn(5), 'logy'),
            # (transforms_v2.RollingLogReturn(20), 'logy20'),
            # (transforms_v2.RollingLogReturn(60), 'logy60'),
            # (transforms_v2.RollingLogReturn(120), 'logy120'),
            (transforms_v2.RollingLogReturn(20), 'logy'),
            (transforms_v2.RollingLogReturn(60), 'logy60'),
            (transforms_v2.RollingLogReturn(120), 'logy120'),
            (transforms_v2.RollingLogReturn(250), 'logy250'),
            (transforms_v2.RollingMeanReturn(250), 'mu'),
            (transforms_v2.RollingStdReturn(250), 'sigma'),
        ])

        self.df, self.columns = transforms_apply.apply(
            self.df, self.columns,
            reduce='concat')

class AssetData(AplusData):
    def __init__(self, file_nm='asset_data_20201201.txt'):
        super(AssetData, self).__init__(file_nm)


class IncomeData(AplusData):
    def __init__(self, file_nm='income_data_20200820.txt'):
        super(IncomeData, self).__init__(file_nm)


class MacroData(DataFromFiles):
    def __init__(self, file_nm='macro_data_20200914.txt'):
        super().__init__(file_nm)
        self.df['copper_gold_r'] = self.df['hg1 comdty'] / self.df['gc1 comdty']
        self.df['spx_dj'] = self.df['spx index'] / self.df['indu index']
        self.df['spx_rs'] = self.df['spx index'] / self.df['rty index']
        self.columns = self.df.columns

    def _transform(self):
        transforms = transforms_v2.Transforms([
            (transforms_v2.RollingNormalize(500), 'normalize'),
        ])

        self.df = transforms.sequential(self.df)


# define Dataset for data_loader
class MultiTaskDatasetForMultiTimesteps(DatasetForTimeSeriesBase):
    """
    MultiTaskDatasetForMultiTimesteps(AplusData())
    [asssigned from Base]
    - data_conf      : arr, idx, columns, label_columns_dict
    - parameter : sampling_days, k_days
    """
    def __init__(self, addible_data, *args, window=250, pos_embedding=True, **kwargs):
        super(MultiTaskDatasetForMultiTimesteps, self).__init__(addible_data, *args, **kwargs)
        self.window = window
        self.pos_embedding = pos_embedding
        self.adj = window % self.sampling_days
        self.default_range = [self.window + self.sampling_days, len(self.idx) - self.k_days]

        if self.pos_embedding:
            self.columns += ['posenc']

    def label_columns_idx(self, key):
        return [self.columns.index(idx) for idx in self.label_columns_dict[key]]

    def add_positional_encoding(self, arr):
        # arr shape: n_timesteps, n_features

        n_timesteps = len(arr)
        timestep_i = np.arange(n_timesteps, dtype=np.float32).reshape([-1, 1]) / n_timesteps
        arr = np.concatenate([arr, timestep_i], axis=-1)

        return arr

    def __getitem__(self, i):
        out = {'features_prev': self.arr[max(0, i - self.window - self.sampling_days):(i + 1 - self.sampling_days)][self.adj::self.sampling_days],
               'features':  self.arr[max(0, i - self.window):(i + 1)][self.adj::self.sampling_days]}

        for key in ['features_prev', 'features']:
            out[key] = self.add_positional_encoding(out[key])

        # for sequential
        labels_prev_base = self.arr[(i + 1 - self.sampling_days):(i + self.k_days + 2 - self.sampling_days)][::self.sampling_days, :]

        # -1 : for spot  (for seq, use ':')
        out['labels_prev'] = dict([(key, labels_prev_base[-1, self.label_columns_idx(key)])
                                   for key in self.label_columns_dict.keys()])

        if i >= self.default_range[1] or i < self.default_range[0]: # very recent data_conf
            out['labels'] = out['labels_prev'] # dummy. not used. 
            # out['labels'] = dict([(key, labels_base[-1, self.label_columns_idx(key)])
            #                       for key in self.label_columns_dict.keys()])
        else:
            labels_base = self.arr[(i+1):(i + self.k_days + 2)][::self.sampling_days, :]
            # print(i, labels_base.shape)
            out['labels'] = dict([(key, labels_base[-1, self.label_columns_idx(key)])
                                  for key in self.label_columns_dict.keys()])

        return out


class DatasetManager(DatasetManagerBase):
    """
        data_list = [AplusData(), MacroData()]
        test_days = 250
        batch_size = 32
        dm = DatasetManager(data_list, test_days, batch_size)
        dl_train = dm.get_data_loader(700, 'train')
        dl_test = dm.get_data_loader(700, 'test')
    """
    def __init__(self, data_list, test_days, batch_size, **kwargs):
        super(DatasetManager, self).__init__(data_list, 'multitask', **kwargs)
        self.test_days = test_days
        self.batch_size = batch_size

    @property
    def labels_list(self):
        return self.dataset.label_columns_dict['logy']

    def define_dataset_func(self, dataset_type):
        if dataset_type in ['multitask']:
            return MultiTaskDatasetForMultiTimesteps

    def mode_params(self, base_i, mode):

        # default range: 전체 데이터 양끝단 못 쓰는 idx 제거 (ie. multistep: [window:-k_days])
        # => 수정: 마지막 k_days는 가능케끔 하고 label=None처리
        default_begin_i, default_end_i = self.dataset.default_range

        if mode == 'train':
            begin_i = 0
            end_i = int(base_i * 0.60) - self.dataset.k_days
            batch_size = self.batch_size
            sampler_type = 'random_sampler'

        elif mode == 'eval':
            begin_i = int(base_i * 0.60) - self.dataset.k_days
            end_i = int(base_i * 0.95) - self.dataset.k_days
            batch_size = self.batch_size
            sampler_type = 'random_without_replacement'

        elif mode == 'test':
            begin_i = base_i
            end_i = min(base_i + self.test_days, len(self.dataset))
            batch_size = self.batch_size # also possible set to len(self.dataset)
            sampler_type = 'sequential_sampler'
            # default_end_i = end_i

        elif mode == 'test_insample':
            begin_i = 0
            end_i = min(base_i + self.test_days, len(self.dataset))
            batch_size = self.batch_size # also possible set to len(self.dataset)
            sampler_type = 'sequential_sampler'

        else:
            raise NotImplementedError

        params = dict(begin_i=max(default_begin_i, begin_i),
                      end_i=end_i,
                      batch_size=batch_size,
                      sampler_type=sampler_type)

        return params

    def get_begin_end_info(self, base_i, mode):
        assert mode in ['test', 'test_insample']
        params_base = self.mode_params(base_i, mode)
        r_ = base_i % self.dataset.k_days
        # time series dataset index info
        if mode == 'test':
            idx_list = [params_base['begin_i'], params_base['end_i']-1]
        elif mode == 'test_insample':
            idx_list = [params_base['begin_i'], int(base_i * 0.9), base_i, params_base['end_i']-1]

        date_ = list(np.array(self.dataset.idx)[idx_list])

        idx_ = [i - idx_list[0] for i in idx_list]  # re-indexing for sub-dataset

        return dict(date_=date_, idx_=idx_)
