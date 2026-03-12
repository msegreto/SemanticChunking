from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.factory import DatasetFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare raw datasets under data/raw/.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, e.g. 'msmarco-docs' or 'beir/scifact'",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional explicit target directory. Defaults to data/raw/<dataset_name>",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only validate local presence; do not download missing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = DatasetFactory.create(args.dataset)

    config = {
        "name": args.dataset,
        "download_if_missing": not args.no_download,
    }
    if args.data_dir:
        config["data_dir"] = args.data_dir

    raw_path = processor.ensure_available(config)
    print(f"[OK] Dataset available at: {raw_path}")


if __name__ == "__main__":
    main()