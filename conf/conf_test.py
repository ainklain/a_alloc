
from omegaconf import DictConfig, OmegaConf
import hydra

from conf.conf_helper import register


def evaluate_cfg(cfg: DictConfig):
    """
    config.yaml에 [eval] 표시가 있으면, eval 함수로 계산
    """
    for key in cfg.keys():
        if type(cfg[key]) == str and '[eval]' in cfg[key]:
            cfg[key] = eval(cfg[key].split('[eval]')[1])
        elif type(cfg[key]) == DictConfig:
            evaluate_cfg(cfg[key])


@hydra.main(config_name="config")
def main(cfg: DictConfig):
    register()
    evaluate_cfg(cfg)

    # print(OmegaConf.to_yaml(cfg))
    print(cfg.model.allocator.in_dim)

    print(type(cfg.model))


if __name__ == "__main__":
    main()