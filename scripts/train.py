"""make train C=configs/train/<recipe>.yaml [SEED=..]"""

import argparse

from tinycast.config import TrainConfig, load
from tinycast.distill.train import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = load(a.config, TrainConfig)
    if a.seed is not None:
        cfg.seed = a.seed
        cfg.name = f"{cfg.name}_s{a.seed}"
    if a.device:
        cfg.device = a.device
    train(cfg)
