import argparse

import _bootstrap
from src.config import load_config
from src.data_generation import generate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_final.yaml")
    parser.add_argument(
        "--start-case-id",
        type=int,
        default=0,
        help="Numeric ID assigned to the first generated case (default: 0).",
    )
    parser.add_argument(
        "--number-of-cases",
        type=int,
        default=None,
        help="Override data_generation.train.number_of_cases from the config.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Fail before generation if any target case file already exists.",
    )
    args = parser.parse_args()
    generate_dataset(
        load_config(args.config),
        split="train",
        start_case_id=args.start_case_id,
        number_of_cases=args.number_of_cases,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
