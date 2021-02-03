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

    tu.set_seed(cfg.experiment.seed)

    ################
    # data
    ################
    data_list = [AplusData(cfg.data.path.asset, base_dir=to_absolute_path('data')),
                 MacroData(cfg.data.path.macro, base_dir=to_absolute_path('data'))]

    print(cfg.keys())
    dm = DatasetManager(data_list, cfg.data.test_days, cfg.data.batch_size)

    ################
    # model
    ################
    try:
        model_cls = model_list[cfg.model.name]
    except KeyError:
        print("possible model name : {}".format(model_list.keys()))
        return

    model = model_cls(model_cfg=cfg.model, exp_cfg=cfg.experiment, dm=dm)

    ################
    # logger
    ################
    logger = TensorBoardLogger(os.path.join(cfg.experiment.outpath, cfg.logger.path_name), name=cfg.logger.name)

    ################
    # trainer
    ################
    trainer_cfg = cfg.trainer

    if trainer_cfg.stage1.use:
        # stage 1
        model.set_stage(1, trainer_cfg)

        early_stop_callback1 = EarlyStopping(
            monitor=trainer_cfg.stage1.monitor,
            min_delta=0.00,
            patience=trainer_cfg.stage1.patience,
            verbose=False,
            mode='min',
        )

        trainer = pl.Trainer(
            gpus=trainer_cfg.gpus,
            check_val_every_n_epoch=trainer_cfg.check_val_every_n_epoch,
            gradient_clip_val=trainer_cfg.gradient_clip_val,
            auto_scale_batch_size=trainer_cfg.auto_scale_batch_size,
            callbacks=[early_stop_callback1],
            logger=logger,
            max_epochs=trainer_cfg.max_epochs,
            # reload_dataloaders_every_epoch=True,
        )
        # trainer.tune(model)

        trainer.fit(model)

        trainer.test(model, ckpt_path='best')

    if trainer_cfg.stage2.use:
        # stage 2
        model.set_stage(2, trainer_cfg)

        early_stop_callback2 = EarlyStopping(
            monitor=trainer_cfg.stage2.monitor,
            min_delta=0.00,
            patience=trainer_cfg.stage2.patience,
            verbose=True,
            mode='min'
        )

        checkpoint_callback2 = ModelCheckpoint(
            monitor=trainer_cfg.stage2.monitor,
            dirpath=cfg.experiment.outpath,
            filename='{model}-{data}-'.format(model=cfg.model.name, data=cfg.data.name,) + "{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode='min',
        )

        trainer = pl.Trainer(
            gpus=trainer_cfg.gpus,
            check_val_every_n_epoch=trainer_cfg.check_val_every_n_epoch,
            gradient_clip_val=trainer_cfg.gradient_clip_val,
            auto_scale_batch_size=trainer_cfg.auto_scale_batch_size,
            callbacks=[early_stop_callback2, checkpoint_callback2],
            logger=logger,
            max_epochs=trainer_cfg.max_epochs,
            # reload_dataloaders_every_epoch=True,
        )

        trainer.fit(model)

        trainer.test(model, ckpt_path='best')

    print("tensorboard --logdir={}".format(os.path.join(cfg.experiment.outpath, cfg.logger.path_name, cfg.logger.name, 'default_0')))


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
        print(OmegaConf.to_yaml(cfg))

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