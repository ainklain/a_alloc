from dataclasses import dataclass
from omegaconf import DictConfig
import hydra
from hydra.core.config_store import ConfigStore
import torch_utils as tu

from v20201222.dataset_v2 import AplusData, MacroData, DatasetManager


@dataclass
class



tu.use_profile()

# @profile
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    base_weight = dict(h=[0.69, 0.2, 0.1, 0.01],
                       m=[0.4, 0.1, 0.075, 0.425],
                       l=[0.25, 0.05, 0.05, 0.65],
                       eq=[0.25, 0.25, 0.25, 0.25])

    data_list = [AplusData('app_data_20201230.txt'), MacroData('macro_data_20210104.txt')]
    dm = DatasetManager(data_list, c.test_days, c.batch_size)

    trainer = Trainer(c, dm)
    trainer.run(ii)

