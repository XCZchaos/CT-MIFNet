"""Command-line options shared by the three CT-MIFNet experiments."""

import argparse
from pathlib import Path


BCI_SUBJECTS = tuple(range(1, 10))


def parse_experiment_args(dataset, *, data_dir, subjects, max_subject, output_dir=None):
    parser = argparse.ArgumentParser(description=f"Train CT-MIFNet on {dataset}.")
    parser.add_argument(
        "--data-dir", type=Path, default=data_dir, metavar="DIR",
        help="Directory containing the prepared subject files.",
    )
    parser.add_argument(
        "--subjects", type=int, nargs="+", choices=range(1, max_subject + 1),
        default=subjects, metavar="ID",
        help="Subject IDs to train, in order (default: %(default)s).",
    )
    if output_dir is not None:
        parser.add_argument(
            "--output-dir", type=Path, default=output_dir, metavar="DIR",
            help="Directory for experiment outputs.",
        )
    return parser.parse_args()
