import argparse

import _bootstrap
from src.config import load_config
from src.data_generation import generate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_final.yaml")
    args = parser.parse_args()
    generate_dataset(load_config(args.config), split="test")


if __name__ == "__main__":
    main()
