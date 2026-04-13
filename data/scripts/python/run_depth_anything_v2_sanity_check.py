#!/usr/bin/env python3

from __future__ import annotations

from common import require_venv_python, resolve_data_dir, run_command


def main() -> None:
    data_dir = resolve_data_dir()
    repo_dir = data_dir / "external" / "Depth-Anything-V2"
    output_dir = data_dir / "outputs" / "sanity_check"
    venv_python = require_venv_python()

    if not repo_dir.exists():
        raise FileNotFoundError(
            "Depth-Anything-V2 repository not found. Run `python data/scripts/slt_data.py bootstrap` first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    run_command(
        [
            str(venv_python),
            "run.py",
            "--encoder",
            "vits",
            "--img-path",
            "assets/examples",
            "--outdir",
            str(output_dir),
            "--pred-only",
        ],
        cwd=repo_dir,
    )

    print()
    print("Sanity check complete.")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
