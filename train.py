import argparse
from pathlib import Path

from lbb2seg.utils.config import load_config
from lbb2seg.training.trainer import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FiLM-SMP flood segmentation model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    train_from_config(cfg)


if __name__ == "__main__":
    main()
