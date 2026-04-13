#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from common import download_file, resolve_root_dir


@dataclass(frozen=True)
class BenchmarkAsset:
    filename: str
    url: str
    expected_size_bytes: int


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    title: str
    assets: tuple[BenchmarkAsset, ...]


BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        key="kitti",
        title="KITTI Depth Prediction",
        assets=(
            BenchmarkAsset(
                filename="data_depth_selection.zip",
                url="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_selection.zip",
                expected_size_bytes=2010655006,
            ),
        ),
    ),
    BenchmarkSpec(
        key="nyu_depth_v2",
        title="NYU Depth V2",
        assets=(
            BenchmarkAsset(
                filename="nyu_depth_v2_labeled.mat",
                url="https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat",
                expected_size_bytes=2972037809,
            ),
        ),
    ),
    BenchmarkSpec(
        key="sintel",
        title="MPI Sintel",
        assets=(
            BenchmarkAsset(
                filename="MPI-Sintel-complete.zip",
                url="https://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip",
                expected_size_bytes=5627783629,
            ),
        ),
    ),
    BenchmarkSpec(
        key="eth3d",
        title="ETH3D",
        assets=(
            BenchmarkAsset(
                filename="multi_view_training_dslr_jpg.7z",
                url="https://www.eth3d.net/data/multi_view_training_dslr_jpg.7z",
                expected_size_bytes=5054974037,
            ),
            BenchmarkAsset(
                filename="multi_view_training_dslr_scan_eval.7z",
                url="https://www.eth3d.net/data/multi_view_training_dslr_scan_eval.7z",
                expected_size_bytes=1920731473,
            ),
        ),
    ),
    BenchmarkSpec(
        key="diode",
        title="DIODE",
        assets=(
            BenchmarkAsset(
                filename="val.tar.gz",
                url="http://diode-dataset.s3.amazonaws.com/val.tar.gz",
                expected_size_bytes=2774625282,
            ),
        ),
    ),
)

MIN_FREE_SPACE_BUFFER_BYTES = 2 * 1024 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the remaining evaluation benchmark archives.")
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=[benchmark.key for benchmark in BENCHMARKS],
        help="Optional subset of benchmark keys to download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force a fresh download even if a valid local file already exists.",
    )
    return parser.parse_args()


def selected_benchmarks(keys: list[str]) -> list[BenchmarkSpec]:
    registry = {benchmark.key: benchmark for benchmark in BENCHMARKS}
    missing = [key for key in keys if key not in registry]
    if missing:
        raise ValueError(f"Unknown benchmark keys: {', '.join(sorted(missing))}")
    return [registry[key] for key in keys]


def ensure_free_disk_space(target_path: Path, expected_size_bytes: int) -> None:
    existing_size = target_path.stat().st_size if target_path.exists() else 0
    remaining_bytes = max(expected_size_bytes - existing_size, 0)
    free_bytes = shutil.disk_usage(target_path.parent).free
    required_bytes = remaining_bytes + MIN_FREE_SPACE_BUFFER_BYTES
    if free_bytes < required_bytes:
        raise RuntimeError(
            f"Not enough free disk space for {target_path.name}: need at least {required_bytes} bytes, have {free_bytes} bytes."
        )


def download_asset(target_path: Path, asset: BenchmarkAsset, force: bool) -> dict[str, int | bool | str]:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_free_disk_space(target_path, asset.expected_size_bytes)
    return download_file(
        asset.url,
        target_path,
        expected_size_bytes=asset.expected_size_bytes,
        force=force,
    )


def write_metadata(dataset_root: Path, benchmark: BenchmarkSpec, results: list[dict[str, int | bool | str]]) -> None:
    metadata = {
        "benchmark": benchmark.title,
        "key": benchmark.key,
        "assets": [
            {
                **asdict(asset),
                **result,
            }
            for asset, result in zip(benchmark.assets, results, strict=True)
        ],
    }
    metadata_path = dataset_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root_dir = resolve_root_dir()
    datasets_root = root_dir / "data" / "datasets"

    summary: dict[str, dict[str, int | str]] = {}
    for benchmark in selected_benchmarks(args.benchmarks):
        dataset_root = datasets_root / benchmark.key
        raw_dir = dataset_root / "raw"
        results = []
        for asset in benchmark.assets:
            results.append(download_asset(raw_dir / asset.filename, asset, force=args.force))

        write_metadata(dataset_root, benchmark, results)
        summary[benchmark.key] = {
            "benchmark": benchmark.title,
            "asset_count": len(results),
            "total_size_bytes": sum(int(result["size_bytes"]) for result in results),
            "raw_dir": str(raw_dir),
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
