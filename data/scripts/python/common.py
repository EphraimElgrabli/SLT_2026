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
PROGRESS_REFRESH_SECONDS = 0.5
BYTES_PER_MB = 1024 * 1024


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


# ---------------------------------------------------------------------------
# Progress reporting helpers
# ---------------------------------------------------------------------------
def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def _format_progress_line(
    name: str,
    downloaded_bytes: int,
    total_bytes: int,
    speed_bytes_per_sec: float,
    eta_seconds: float | None,
) -> str:
    downloaded_mb = downloaded_bytes / BYTES_PER_MB
    speed_mb = speed_bytes_per_sec / BYTES_PER_MB
    if total_bytes > 0:
        total_mb = total_bytes / BYTES_PER_MB
        percent = downloaded_bytes / total_bytes * 100
        eta_text = f"ETA {_format_duration(eta_seconds)}" if eta_seconds is not None else "ETA --"
        return (
            f"  {name}  {downloaded_mb:8.1f}/{total_mb:8.1f} MB  "
            f"{percent:5.1f}%  {speed_mb:5.2f} MB/s  {eta_text}"
        )
    return f"  {name}  {downloaded_mb:8.1f} MB  {speed_mb:5.2f} MB/s"


def _stream_to_file_with_progress(
    response,
    file_handle,
    *,
    name: str,
    initial_size: int,
    total_size: int,
) -> None:
    downloaded = initial_size
    start_time = time.monotonic()
    next_refresh = start_time

    # Print one initial line so the user immediately sees something happening,
    # even before the first chunk arrives.
    sys.stdout.write(_format_progress_line(name, downloaded, total_size, 0.0, None))
    sys.stdout.flush()

    try:
        while True:
            chunk = response.read(DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            file_handle.write(chunk)
            downloaded += len(chunk)

            now = time.monotonic()
            if now < next_refresh:
                continue
            next_refresh = now + PROGRESS_REFRESH_SECONDS

            elapsed = now - start_time
            session_bytes = downloaded - initial_size
            speed = session_bytes / elapsed if elapsed > 0 else 0.0
            eta = None
            if total_size > 0 and speed > 0:
                eta = max(total_size - downloaded, 0) / speed

            sys.stdout.write("\r" + _format_progress_line(name, downloaded, total_size, speed, eta))
            sys.stdout.flush()
    finally:
        # Final summary on its own line, regardless of how the loop exited.
        elapsed = max(time.monotonic() - start_time, 1e-6)
        session_bytes = max(downloaded - initial_size, 0)
        avg_speed = session_bytes / elapsed
        downloaded_mb = downloaded / BYTES_PER_MB
        avg_speed_mb = avg_speed / BYTES_PER_MB
        if total_size > 0:
            total_mb = total_size / BYTES_PER_MB
            percent = downloaded / total_size * 100
            summary = (
                f"  {name}  {downloaded_mb:8.1f}/{total_mb:8.1f} MB  "
                f"{percent:5.1f}%  avg {avg_speed_mb:5.2f} MB/s  "
                f"in {_format_duration(elapsed)}"
            )
        else:
            summary = (
                f"  {name}  {downloaded_mb:8.1f} MB  "
                f"avg {avg_speed_mb:5.2f} MB/s  in {_format_duration(elapsed)}"
            )
        sys.stdout.write("\r" + summary + "\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Download primitives
# ---------------------------------------------------------------------------
def _open_download(url: str, existing_size: int):
    headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}
    request = Request(url, headers=headers)
    return urlopen(request)


def _resolve_total_size(response, *, existing_size: int, is_resuming: bool) -> int:
    content_length_header = response.headers.get("Content-Length")
    content_length = int(content_length_header) if content_length_header else 0
    if is_resuming:
        return existing_size + content_length
    return content_length


def _download_with_resume(url: str, target_path: Path) -> None:
    existing_size = target_path.stat().st_size if target_path.exists() else 0
    name = target_path.name

    try:
        response = _open_download(url, existing_size)
    except HTTPError as error:
        if error.code == 416:
            # The server reports the range is past the end of the file, which means
            # the local file is already at or beyond the full remote size.
            return
        raise

    status_code = getattr(response, "status", None)
    is_resuming = existing_size > 0 and status_code == 206

    if existing_size > 0 and not is_resuming:
        # Server ignored our Range header; restart cleanly from byte zero.
        response.close()
        target_path.unlink(missing_ok=True)
        existing_size = 0
        response = _open_download(url, 0)

    total_size = _resolve_total_size(response, existing_size=existing_size, is_resuming=is_resuming)
    if is_resuming and existing_size > 0:
        print(
            f"  {name}  resuming from {existing_size / BYTES_PER_MB:.1f} MB "
            f"({existing_size / total_size * 100:.1f}% already on disk)"
            if total_size > 0
            else f"  {name}  resuming from {existing_size / BYTES_PER_MB:.1f} MB"
        )
    else:
        size_text = f"{total_size / BYTES_PER_MB:.1f} MB" if total_size > 0 else "size unknown"
        print(f"  {name}  starting download ({size_text})")

    mode = "ab" if is_resuming else "wb"

    initial_size = existing_size if is_resuming else 0
    with response, target_path.open(mode) as file_handle:
        _stream_to_file_with_progress(
            response,
            file_handle,
            name=name,
            initial_size=initial_size,
            total_size=total_size,
        )


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
        print(f"  {target_path.name}  already complete ({existing_size / BYTES_PER_MB:.1f} MB), skipping.")
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
            wait_seconds = attempt
            print(f"  {target_path.name}  attempt {attempt} failed ({error}); retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)
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