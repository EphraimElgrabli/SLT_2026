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
        help="Clone the pinned upstream repo, create the local venv, and fetch the small checkpoint.",
    )
    bootstrap_parser.add_argument(
        "--force-checkpoint",
        action="store_true",
        help="Re-download the small checkpoint even if it already exists.",
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
        help="Re-download the small checkpoint during bootstrap.",
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

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
