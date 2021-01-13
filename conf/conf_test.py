
from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main( config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(cfg.data.outpath)
    print(cfg.data.data_path.asset)


if __name__ == "__main__":
    main()