
from omegaconf import OmegaConf, DictConfig


def register():
    OmegaConf.register_resolver("concat", lambda x, y: x+y)
    OmegaConf.register_resolver("add", lambda x, y: int(x) + int(y))
    OmegaConf.register_resolver("sub", lambda x, y: int(x) - int(y))
    OmegaConf.register_resolver("mul", lambda x, y: int(x) * int(y))
    OmegaConf.register_resolver("div", lambda x, y: int(x) // int(y))


def evaluate_cfg(cfg: DictConfig):
    """
    config.yaml에 (eval) 표시가 있으면, eval 함수로 계산
    """
    for key in cfg.keys():
        if type(cfg[key]) == str and '(eval)' in cfg[key]:
            cfg[key] = eval(cfg[key].split('(eval)')[-1])
        elif type(cfg[key]) == DictConfig:
            evaluate_cfg(cfg[key])
