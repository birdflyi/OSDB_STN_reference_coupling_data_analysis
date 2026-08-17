"""Frozen-run provenance and checksum helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import GH_CoRE


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path, base: str | Path | None = None) -> dict[str, Any]:
    item = Path(path).resolve()
    name = str(item.relative_to(Path(base).resolve())) if base else str(item)
    return {"path": name.replace("\\", "/"), "bytes": item.stat().st_size, "sha256": sha256_file(item)}


def file_records(paths: Iterable[str | Path], base: str | Path | None = None) -> list[dict[str, Any]]:
    return [file_record(path, base) for path in sorted(map(Path, paths), key=lambda value: str(value).lower())]


def git_info(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    return {
        "branch": _git(["branch", "--show-current"], root),
        "commit": _git(["rev-parse", "HEAD"], root),
        "status_short": _git(["status", "--short"], root).splitlines(),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def runtime_versions() -> dict[str, str]:
    import networkx
    import numpy
    import pandas
    import scipy

    return {
        "python": platform.python_version(),
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "networkx": networkx.__version__,
        "gh_core": getattr(GH_CoRE, "__version__", "unknown"),
    }


def _git(args: list[str], root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=str(root), check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()
