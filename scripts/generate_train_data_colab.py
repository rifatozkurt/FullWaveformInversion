import argparse

import _bootstrap
from src.config import load_config
from src.data_generation_colab import generate_dataset_colab


def main():
    parser = argparse.ArgumentParser(
        description="Experimental multi-case GPU-batched training-data generator."
    )
    parser.add_argument("--config", default="configs/experimental.yaml")
    parser.add_argument("--start-case-id", type=int, default=0)
    parser.add_argument("--number-of-cases", type=int, default=None)
    parser.add_argument("--case-batch-size", type=int, default=2)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-overwrite", action="store_true")
    args = parser.parse_args()
    generate_dataset_colab(
        load_config(args.config),
        split="train",
        output_dir=args.output_dir,
        start_case_id=args.start_case_id,
        number_of_cases=args.number_of_cases,
        case_batch_size=args.case_batch_size,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
