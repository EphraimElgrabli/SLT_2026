#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def scripts_root() -> Path:
    return Path(__file__).resolve().parent


def python_tasks_dir() -> Path:
    return scripts_root() / "python"


def venv_python() -> Path:
    root_dir = scripts_root().parents[1]
    venv_dir = root_dir / "data" / ".venv"
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run_python_script(script_name: str, *script_args: str, use_venv: bool = False) -> None:
    python_executable = venv_python() if use_venv else Path(sys.executable)
    if use_venv and not python_executable.exists():
        raise FileNotFoundError(
            "Virtual environment not found. Run `python data/scripts/slt_data.py bootstrap` first."
        )

    subprocess.run(
        [str(python_executable), str(python_tasks_dir() / script_name), *script_args],
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master workflow runner for the Depth Anything V2 reproduction data scripts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        help="Clone the pinned upstream repo, create the local venv, and fetch the public student checkpoints.",
    )
    bootstrap_parser.add_argument(
        "--force-checkpoint",
        action="store_true",
        help="Re-download the public student checkpoints even if they already exist.",
    )

    subparsers.add_parser(
        "sanity-check",
        help="Run the Depth Anything V2 smoke test against the bundled example images.",
    )

    da2k_parser = subparsers.add_parser(
        "da2k",
        help="Download and preprocess the DA-2K benchmark dataset.",
    )
    da2k_parser.add_argument(
        "--force",
        action="store_true",
        help="Force a fresh DA-2K download and rebuild the processed manifests.",
    )

    benchmarks_parser = subparsers.add_parser(
        "benchmarks",
        help="Download the remaining paper evaluation benchmark archives.",
    )
    benchmarks_parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Optional subset of benchmarks to download.",
    )
    benchmarks_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download assets even if the local archive size already matches expectations.",
    )

    setup_parser = subparsers.add_parser(
        "setup",
        help="Run `bootstrap` followed by `sanity-check` as the default machine bootstrap flow.",
    )
    setup_parser.add_argument(
        "--force-checkpoint",
        action="store_true",
        help="Re-download the public student checkpoints during bootstrap.",
    )

    evaluate_da2k_parser = subparsers.add_parser(
        "evaluate-da2k",
        help="Run the Depth Anything V2 student models on the DA-2K benchmark and write accuracy results.",
    )
    evaluate_da2k_parser.add_argument(
        "--models",
        nargs="+",
        choices=["vits", "vitb", "vitl"],
        default=None,
        help="Encoders to evaluate (default: all three).",
    )
    evaluate_da2k_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N images. Useful for quick sanity checks.",
    )
    evaluate_da2k_parser.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="Inference resolution (default 518, matches the paper).",
    )
    evaluate_da2k_parser.add_argument(
        "--force",
        action="store_true",
        help="Discard existing predictions for the selected models and re-run from scratch.",
    )

    metric_checkpoints_parser = subparsers.add_parser(
        "metric-checkpoints",
        help="Download the small indoor/outdoor metric-depth checkpoints used by benchmark evaluation.",
    )
    metric_checkpoints_parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Optional subset: hypersim_vits and/or vkitti_vits.",
    )
    metric_checkpoints_parser.add_argument("--force", action="store_true", help="Re-download checkpoint files.")

    evaluate_metric_parser = subparsers.add_parser(
        "evaluate-metric",
        help="Run metric-depth evaluation on local KITTI, NYU, Sintel, DIODE, and ETH3D assets.",
    )
    evaluate_metric_parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["kitti", "nyu_depth_v2", "sintel", "eth3d", "diode"],
        default=None,
        help="Optional dataset subset to evaluate.",
    )
    evaluate_metric_parser.add_argument("--limit", type=int, default=None, help="Optional per-dataset sample limit.")
    evaluate_metric_parser.add_argument("--input-size", type=int, default=None, help="Inference resolution.")
    evaluate_metric_parser.add_argument(
        "--median-align",
        action="store_true",
        help="Apply per-image median scaling before metric calculation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "bootstrap":
        extra_args = ["--force-checkpoint"] if args.force_checkpoint else []
        run_python_script("bootstrap_depth_anything_v2.py", *extra_args)
        return

    if args.command == "sanity-check":
        run_python_script("run_depth_anything_v2_sanity_check.py")
        return

    if args.command == "da2k":
        workflow_args = ["--force"] if args.force else []
        run_python_script("acquire_da2k.py", *workflow_args, use_venv=True)
        run_python_script("prepare_da2k.py", *workflow_args, use_venv=True)
        return

    if args.command == "benchmarks":
        workflow_args: list[str] = []
        if args.force:
            workflow_args.append("--force")
        if args.benchmarks:
            workflow_args.extend(["--benchmarks", *args.benchmarks])
        run_python_script("acquire_remaining_benchmarks.py", *workflow_args)
        return

    if args.command == "setup":
        extra_args = ["--force-checkpoint"] if args.force_checkpoint else []
        run_python_script("bootstrap_depth_anything_v2.py", *extra_args)
        run_python_script("run_depth_anything_v2_sanity_check.py")
        return

    if args.command == "evaluate-da2k":
        workflow_args = []
        if args.models:
            workflow_args.extend(["--models", *args.models])
        if args.limit is not None:
            workflow_args.extend(["--limit", str(args.limit)])
        if args.input_size is not None:
            workflow_args.extend(["--input-size", str(args.input_size)])
        if args.force:
            workflow_args.append("--force")
        run_python_script("evaluate_da2k.py", *workflow_args, use_venv=True)
        return

    if args.command == "metric-checkpoints":
        workflow_args = []
        if args.force:
            workflow_args.append("--force")
        if args.checkpoints:
            workflow_args.extend(["--checkpoints", *args.checkpoints])
        run_python_script("acquire_metric_checkpoints.py", *workflow_args)
        return

    if args.command == "evaluate-metric":
        workflow_args = []
        if args.datasets:
            workflow_args.extend(["--datasets", *args.datasets])
        if args.limit is not None:
            workflow_args.extend(["--limit", str(args.limit)])
        if args.input_size is not None:
            workflow_args.extend(["--input-size", str(args.input_size)])
        if args.median_align:
            workflow_args.append("--median-align")
        run_python_script("evaluate_metric_depth.py", *workflow_args, use_venv=True)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
