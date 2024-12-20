import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="arp")
def main(cfg: DictConfig):
    # Convert the configuration to a dictionary and resolve interpolations
    print(OmegaConf.to_container(cfg, resolve=True))

if __name__ == "__main__":
    main()