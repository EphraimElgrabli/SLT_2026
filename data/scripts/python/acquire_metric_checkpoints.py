#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass

from common import download_file, resolve_data_dir, sync_file_link


@dataclass(frozen=True)
class MetricCheckpointSpec:
    key: str
    scene: str
    encoder: str
    filename: str
    repo_id: str
    max_depth: float


METRIC_CHECKPOINTS: tuple[MetricCheckpointSpec, ...] = (
    MetricCheckpointSpec(
        key="hypersim_vits",
        scene="indoor",
        encoder="vits",
        filename="depth_anything_v2_metric_hypersim_vits.pth",
        repo_id="depth-anything/Depth-Anything-V2-Metric-Hypersim-Small",
        max_depth=20.0,
    ),
    MetricCheckpointSpec(
        key="vkitti_vits",
        scene="outdoor",
        encoder="vits",
        filename="depth_anything_v2_metric_vkitti_vits.pth",
        repo_id="depth-anything/Depth-Anything-V2-Metric-VKITTI-Small",
        max_depth=80.0,
    ),
)


def checkpoint_url(spec: MetricCheckpointSpec) -> str:
    return f"https://huggingface.co/{spec.repo_id}/resolve/main/{spec.filename}?download=true"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Depth Anything V2 metric-depth checkpoints.")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=[spec.key for spec in METRIC_CHECKPOINTS],
        help="Metric checkpoint keys to download.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download checkpoints even when present.")
    return parser.parse_args()


def selected_specs(keys: list[str]) -> list[MetricCheckpointSpec]:
    registry = {spec.key: spec for spec in METRIC_CHECKPOINTS}
    missing = [key for key in keys if key not in registry]
    if missing:
        raise ValueError(f"Unknown metric checkpoint keys: {', '.join(sorted(missing))}")
    return [registry[key] for key in keys]


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir()
    models_dir = data_dir / "models"
    metric_repo_checkpoints = data_dir / "external" / "Depth-Anything-V2" / "metric_depth" / "checkpoints"
    models_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for spec in selected_specs(args.checkpoints):
        target_path = models_dir / spec.filename
        result = download_file(checkpoint_url(spec), target_path, force=args.force)
        sync_file_link(target_path, metric_repo_checkpoints / spec.filename)
        summary[spec.key] = {
            "scene": spec.scene,
            "encoder": spec.encoder,
            "max_depth": spec.max_depth,
            **result,
        }

    for key, result in summary.items():
        print(f"{key}: {result['path']} ({result['size_bytes']} bytes)")


if __name__ == "__main__":
    main()
