#!/usr/bin/env python3

"""Download the NYU Depth V2 validation split (654-image Eigen test set).

Source: the `sayakpaul/nyu_depth_v2` dataset on the HuggingFace Hub,
auto-converted to parquet. We fetch only the 'validation' split, which is
the standard NYU-D test set used by MiDaS, ZoeDepth, and Depth Anything
for relative-depth evaluation. The much larger 'train' split is not needed.

The two validation parquet shards total ~1 GB. download_file() resumes
partial downloads and verifies the final size against expected_size_bytes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from common import download_file, resolve_root_dir


@dataclass(frozen=True)
class ParquetShard:
    filename: str
    url: str
    expected_size_bytes: int


# URLs come from the HuggingFace datasets-server parquet listing for
# sayakpaul/nyu_depth_v2, split=validation. They live on the special
# refs/convert/parquet git ref (the auto-converted parquet branch).
VALIDATION_SHARDS: tuple[ParquetShard, ...] = (
    ParquetShard(
        filename="validation-0000.parquet",
        url=(
            "https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/"
            "refs%2Fconvert%2Fparquet/default/partial-validation/0000.parquet"
        ),
        expected_size_bytes=634487865,
    ),
    ParquetShard(
        filename="validation-0001.parquet",
        url=(
            "https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/"
            "refs%2Fconvert%2Fparquet/default/partial-validation/0001.parquet"
        ),
        expected_size_bytes=406412648,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the NYU Depth V2 validation parquet shards.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the local file size already matches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = resolve_root_dir()
    raw_dir = root_dir / "data" / "datasets" / "nyu_depth_v2" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for shard in VALIDATION_SHARDS:
        result = download_file(
            shard.url,
            raw_dir / shard.filename,
            expected_size_bytes=shard.expected_size_bytes,
            force=args.force,
        )
        results.append(result)

    print(json.dumps({"validation_shards": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()