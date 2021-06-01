import os
from omegaconf import DictConfig
import hydra
from hydra.utils import to_absolute_path
import sys
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, GPUStatsMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from conf.conf_helper import evaluate_cfg
from dataprocess.dataset_v2 import DatasetManager, AplusData, MacroData, IncomeData, AssetData, DummyMacroData
from models import model_list
import torch_utils as tu

tu.use_profile()


# @hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # initialize cfg
    evaluate_cfg(cfg)
    data_cfg = cfg.data
    model_cfg = cfg.model
    trainer_cfg = cfg.trainer
    exp_cfg = cfg.experiment
    logger_cfg = cfg.logger

    tu.set_seed(exp_cfg.seed)

    ################
    # data
    ################

    data_list = [AplusData(data_cfg.path.asset,
                           base_dir=to_absolute_path('data'),
                           excluded_col_idx=exp_cfg.excluded_asset_idx),
                 MacroData(data_cfg.path.macro,
                           base_dir=to_absolute_path('data'))]

    # 제외 자산이 있는 경우
    if exp_cfg.excluded_asset_idx is not None:
        exp_cfg.excluded_asset_idx.reverse()   # 2개 이상일때 뒤에서부터 pop해야 안꼬임
        for exc_i in exp_cfg.excluded_asset_idx:
            exp_cfg.base_weight.pop(exc_i)
            exp_cfg.wgt_range_min.pop(exc_i)
            exp_cfg.wgt_range_max.pop(exc_i)

        # 자산 제외 후
        exp_cfg.base_weight = [w / sum(exp_cfg.base_weight) for w in exp_cfg.base_weight]

    dm = DatasetManager(data_list, data_cfg.test_days, data_cfg.batch_size, data_cfg.eval_rate)

    ################
    # model
    ################
    try:
        model_cls = model_list[model_cfg.name]
    except KeyError:
        print("possible model name : {}".format(model_list.keys()))
        return

    model = model_cls(model_cfg=model_cfg, exp_cfg=exp_cfg, dm=dm)

    ################
    # logger
    ################
    logger = TensorBoardLogger(os.path.join(exp_cfg.outpath, logger_cfg.path_name), name=logger_cfg.name)
    print("tensorboard --logdir={}".format(os.path.join(os.path.abspath(exp_cfg.outpath), logger_cfg.path_name, logger_cfg.name, 'version_0')))

    ################
    # trainer
    ################
    if trainer_cfg.stage1.use:
        # stage 1
        model.set_stage(1, trainer_cfg)

        checkpoint_callback1 = ModelCheckpoint(
            monitor=trainer_cfg.stage1.monitor,
            dirpath=exp_cfg.outpath,
            filename='stage1-{model}-{data}-'.format(model=model_cfg.name, data=data_cfg.name,) + "{epoch:02d}-{val_loss:.2f}",
            save_top_k=2,
            mode='min',
        )

        trainer = pl.Trainer(
            gpus=trainer_cfg.gpus,
            check_val_every_n_epoch=trainer_cfg.check_val_every_n_epoch,
            gradient_clip_val=trainer_cfg.gradient_clip_val,
            auto_scale_batch_size=trainer_cfg.auto_scale_batch_size,
            callbacks=[checkpoint_callback1],
            # logger=logger,
            max_epochs=trainer_cfg.max_epochs,
            # reload_dataloaders_every_epoch=True,
        )
        # trainer.tune(model)

        trainer.fit(model)


    if trainer_cfg.stage2.use:
        # stage 2
        model.set_stage(2, trainer_cfg)


    print("tensorboard --logdir={}".format(os.path.join(os.path.abspath(exp_cfg.outpath), logger_cfg.path_name, logger_cfg.name, 'version_0')))


def main_wrapper():
    if hasattr(sys, 'ps1'):
        """
        REPL Mode
        """

        from hydra.core.global_hydra import GlobalHydra
        from hydra.experimental import compose, initialize
        from omegaconf import OmegaConf

        GlobalHydra.instance().clear()
        initialize(config_path="./conf", job_name="test_app")
        cfg = compose(config_name="config")
        # print(OmegaConf.to_yaml(cfg))

        cfg.experiment.outpath = "./out/test_app/"  # TODO: 임시... hydra.run.dir에 접근하는 방법 못찾음
        os.makedirs(cfg.experiment.outpath, exist_ok=True)

        cfg.model.name = 'epoch_adjust'
        main(cfg)

    else:
        """
        CLI Mode
        """

        @hydra.main(config_path="../conf", config_name="config")
        def func(cfg: DictConfig):
            return main(cfg)

        func()


if __name__ == '__main__':
    main_wrapper()


# python -m v3_latest.main_pl_v1 --multirun exp_conf@experiment=l,m,h,eq