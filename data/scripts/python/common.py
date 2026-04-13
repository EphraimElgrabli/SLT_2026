#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists() and (candidate / "data").exists():
            return candidate

    raise RuntimeError(f"Could not locate the repository root from {start}.")


def resolve_root_dir() -> Path:
    return find_repo_root(Path(__file__).resolve().parent)


def resolve_data_dir() -> Path:
    return resolve_root_dir() / "data"


def resolve_venv_dir() -> Path:
    return resolve_data_dir() / ".venv"


def resolve_venv_python(venv_dir: Path | None = None) -> Path:
    current_venv_dir = venv_dir or resolve_venv_dir()
    relative_path = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    return current_venv_dir / relative_path


def require_venv_python() -> Path:
    python_path = resolve_venv_python()
    if not python_path.exists():
        raise FileNotFoundError(
            "Virtual environment not found. Run `python data/scripts/slt_data.py bootstrap` first."
        )

    return python_path


def run_command(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def sync_file_link(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.is_symlink():
        if target_path.resolve() == source_path.resolve():
            return
        target_path.unlink()
    elif target_path.exists():
        try:
            if target_path.samefile(source_path):
                return
        except FileNotFoundError:
            pass
        target_path.unlink()

    try:
        target_path.symlink_to(source_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def _open_download(url: str, existing_size: int):
    headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}
    request = Request(url, headers=headers)
    return urlopen(request)


def _download_with_resume(url: str, target_path: Path) -> None:
    existing_size = target_path.stat().st_size if target_path.exists() else 0

    try:
        response = _open_download(url, existing_size)
    except HTTPError as error:
        if error.code == 416:
            return
        raise

    with response:
        status_code = getattr(response, "status", None)
        can_resume = existing_size > 0 and status_code == 206

        if existing_size > 0 and not can_resume:
            target_path.unlink(missing_ok=True)
            existing_size = 0

    response = _open_download(url, existing_size)
    mode = "ab" if existing_size > 0 else "wb"

    with response, target_path.open(mode) as file_handle:
        shutil.copyfileobj(response, file_handle, length=DOWNLOAD_CHUNK_SIZE)


def download_file(
    url: str,
    target_path: Path,
    *,
    expected_size_bytes: int | None = None,
    force: bool = False,
    retries: int = 3,
) -> dict[str, int | bool | str]:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if force and target_path.exists():
        target_path.unlink()

    existing_size = target_path.stat().st_size if target_path.exists() else 0
    if expected_size_bytes is not None and existing_size == expected_size_bytes:
        return {
            "path": str(target_path),
            "size_bytes": existing_size,
            "skipped": True,
        }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _download_with_resume(url, target_path)
            break
        except (HTTPError, URLError, OSError) as error:
            last_error = error
            if attempt == retries:
                raise
            time.sleep(attempt)
    else:
        raise RuntimeError(f"Failed to download {url}") from last_error

    actual_size = target_path.stat().st_size
    if expected_size_bytes is not None and actual_size != expected_size_bytes:
        raise RuntimeError(
            f"Downloaded size mismatch for {target_path.name}: expected {expected_size_bytes}, got {actual_size}"
        )

    return {
        "path": str(target_path),
        "size_bytes": actual_size,
        "skipped": False,
    }


def python_command() -> str:
    return sys.executable
