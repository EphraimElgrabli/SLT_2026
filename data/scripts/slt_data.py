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


def flag_args(args: argparse.Namespace, *names: str) -> list[str]:
    """Convert true argparse flags to command-line arguments."""
    return [f"--{name.replace('_', '-')}" for name in names if getattr(args, name)]


def optional_value_args(args: argparse.Namespace, *names: str) -> list[str]:
    """Convert non-null argparse values to command-line arguments."""
    result: list[str] = []
    for name in names:
        value = getattr(args, name)
        if value is not None:
            result.extend([f"--{name.replace('_', '-')}", str(value)])
    return result


def optional_list_args(args: argparse.Namespace, *names: str) -> list[str]:
    """Convert non-empty argparse lists to command-line arguments."""
    result: list[str] = []
    for name in names:
        values = getattr(args, name)
        if values:
            result.extend([f"--{name.replace('_', '-')}", *values])
    return result


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

    prepare_diode_parser = subparsers.add_parser(
        "prepare-diode",
        help="Extract and validate the DIODE val benchmark, writing samples.jsonl.",
    )
    prepare_diode_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract and rebuild the processed manifest.",
    )

    prepare_sintel_parser = subparsers.add_parser(
        "prepare-sintel",
        help="Extract and validate the MPI-Sintel depth benchmark, writing samples.jsonl.",
    )
    prepare_sintel_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract and rebuild the processed manifest.",
    )

    prepare_eth3d_parser = subparsers.add_parser(
        "prepare-eth3d",
        help="Extract ETH3D and build dense GT depth maps by projecting scan PLYs onto DSLR images.",
    )
    prepare_eth3d_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract archives and recompute all depth maps from scratch.",
    )

    acquire_nyu_parser = subparsers.add_parser(
        "acquire-nyu",
        help="Download the NYU Depth V2 validation split (654-image test set) parquet shards.",
    )
    acquire_nyu_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if local file sizes already match.",
    )

    prepare_nyu_parser = subparsers.add_parser(
        "prepare-nyu",
        help="Decode the NYU Depth V2 validation parquet into RGB+depth files and samples.jsonl.",
    )
    prepare_nyu_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-decode even if outputs already exist.",
    )

    acquire_kitti_parser = subparsers.add_parser(
        "acquire-kitti",
        help="Download KITTI Eigen-val raw drives + data_depth_annotated GT depth.",
    )
    acquire_kitti_parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    acquire_kitti_parser.add_argument("--skip-annotated", action="store_true", help="Skip the 13.3 GB GT zip.")
    acquire_kitti_parser.add_argument("--skip-drives", action="store_true", help="Skip per-drive raw downloads.")

    prepare_kitti_parser = subparsers.add_parser(
        "prepare-kitti",
        help="Extract KITTI image_02 + val GT depth and build samples.jsonl.",
    )
    prepare_kitti_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract everything and rebuild the manifest.",
    )

    eval_pw_parser = subparsers.add_parser(
        "evaluate-pixelwise",
        help="Affine-invariant relative-depth evaluation (Table 2).",
    )
    eval_pw_parser.add_argument("--benchmark", default="all",
        choices=["nyu_depth_v2", "kitti", "sintel", "eth3d", "diode", "all"])
    eval_pw_parser.add_argument("--models", nargs="+", default=None,
        choices=["vits", "vitb", "vitl"])
    eval_pw_parser.add_argument("--limit", type=int, default=None)
    eval_pw_parser.add_argument("--threads", type=int, default=4)
    eval_pw_parser.add_argument("--force", action="store_true")

    report_pw_parser = subparsers.add_parser(
        "build-pixelwise-report",
        help="Generate the Table-2 reproduction Word report.",
    )
    report_pw_parser.add_argument("--output", default=None, help="Output .docx path.")

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
        extra_args = flag_args(args, "force_checkpoint")
        run_python_script("bootstrap_depth_anything_v2.py", *extra_args)
        return

    if args.command == "sanity-check":
        run_python_script("run_depth_anything_v2_sanity_check.py")
        return

    if args.command == "da2k":
        workflow_args = flag_args(args, "force")
        run_python_script("acquire_da2k.py", *workflow_args, use_venv=True)
        run_python_script("prepare_da2k.py", *workflow_args, use_venv=True)
        return

    if args.command == "benchmarks":
        workflow_args = flag_args(args, "force")
        workflow_args.extend(optional_list_args(args, "benchmarks"))
        run_python_script("acquire_remaining_benchmarks.py", *workflow_args)
        return

    if args.command == "setup":
        extra_args = flag_args(args, "force_checkpoint")
        run_python_script("bootstrap_depth_anything_v2.py", *extra_args)
        run_python_script("run_depth_anything_v2_sanity_check.py")
        return

    if args.command == "evaluate-da2k":
        workflow_args = optional_list_args(args, "models")
        workflow_args.extend(optional_value_args(args, "limit", "input_size"))
        workflow_args.extend(flag_args(args, "force"))
        run_python_script("evaluate_da2k.py", *workflow_args, use_venv=True)
        return

    if args.command == "prepare-diode":
        workflow_args = flag_args(args, "force")
        run_python_script("prepare_diode.py", *workflow_args, use_venv=True)
        return

    if args.command == "prepare-sintel":
        workflow_args = flag_args(args, "force")
        run_python_script("prepare_sintel.py", *workflow_args, use_venv=True)
        return

    if args.command == "prepare-eth3d":
        workflow_args = flag_args(args, "force")
        run_python_script("prepare_eth3d.py", *workflow_args, use_venv=True)
        return

    if args.command == "acquire-nyu":
        workflow_args = flag_args(args, "force")
        run_python_script("acquire_nyu.py", *workflow_args, use_venv=True)
        return

    if args.command == "prepare-nyu":
        workflow_args = flag_args(args, "force")
        run_python_script("prepare_nyu.py", *workflow_args, use_venv=True)
        return

    if args.command == "acquire-kitti":
        workflow_args = flag_args(args, "force", "skip_annotated", "skip_drives")
        run_python_script("acquire_kitti.py", *workflow_args, use_venv=True)
        return

    if args.command == "prepare-kitti":
        workflow_args = flag_args(args, "force")
        run_python_script("prepare_kitti.py", *workflow_args, use_venv=True)
        return

    if args.command == "evaluate-pixelwise":
        workflow_args = ["--benchmark", args.benchmark, "--threads", str(args.threads)]
        workflow_args.extend(optional_list_args(args, "models"))
        workflow_args.extend(optional_value_args(args, "limit"))
        workflow_args.extend(flag_args(args, "force"))
        run_python_script("evaluate_pixelwise.py", *workflow_args, use_venv=True)
        return

    if args.command == "build-pixelwise-report":
        workflow_args = ["--output", args.output] if args.output else []
        run_python_script("build_pixelwise_report.py", *workflow_args, use_venv=True)
        return

    if args.command == "metric-checkpoints":
        workflow_args = flag_args(args, "force")
        workflow_args.extend(optional_list_args(args, "checkpoints"))
        run_python_script("acquire_metric_checkpoints.py", *workflow_args)
        return

    if args.command == "evaluate-metric":
        workflow_args = optional_list_args(args, "datasets")
        workflow_args.extend(optional_value_args(args, "limit", "input_size"))
        workflow_args.extend(flag_args(args, "median_align"))
        run_python_script("evaluate_metric_depth.py", *workflow_args, use_venv=True)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
