"""Run manifest helpers for isolated Chapter 5 outputs."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .config import Ch5Config


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_info(repo_root: str | Path = ".") -> Dict[str, Any]:
    root = Path(repo_root)
    branch = _git(["branch", "--show-current"], root)
    commit = _git(["rev-parse", "HEAD"], root)
    status = _git(["status", "--short"], root)
    return {
        "branch": branch,
        "commit": commit,
        "short_commit": commit[:7] if commit else "unknown",
        "dirty": bool(status.strip()),
        "status_short": status.splitlines(),
    }


def build_run_id(config: Ch5Config, git_short_commit: str = "unknown", now: Optional[datetime] = None) -> str:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{config.run_id_prefix}_{timestamp}_{git_short_commit}_prep"


def eligible_for_freeze(
    run_successful: bool,
    validation_passed: bool,
    git_commit_exists: bool,
    git_commit_pushed: bool,
    working_tree_policy_satisfied: bool,
) -> bool:
    return (
        run_successful
        and validation_passed
        and git_commit_exists
        and git_commit_pushed
        and working_tree_policy_satisfied
    )


def build_run_manifest(
    config: Ch5Config,
    run_id: str,
    repo_root: str | Path = ".",
    run_status: str = "dry_run",
    validation_passed: bool = False,
    git_commit_pushed: bool = False,
) -> Dict[str, Any]:
    info = git_info(repo_root)
    commit_exists = bool(info.get("commit"))
    freeze_allowed = eligible_for_freeze(
        run_successful=run_status == "success",
        validation_passed=validation_passed,
        git_commit_exists=commit_exists,
        git_commit_pushed=git_commit_pushed,
        working_tree_policy_satisfied=not info.get("dirty", True),
    )
    config_hash = sha256_file(config.path) if config.path else None
    return {
        "run_id": run_id,
        "run_status": run_status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": info,
        "python": platform.python_version(),
        "config_path": str(config.path) if config.path else None,
        "config_sha256": config_hash,
        "random_seed": config.random_seed,
        "planned_outputs": config.planned_outputs(run_id),
        "git_commit_pushed": git_commit_pushed,
        "validation_passed": validation_passed,
        "eligible_for_freeze": freeze_allowed,
    }


def input_manifest_rows(config: Ch5Config, repo_root: str | Path = ".") -> list[Dict[str, Any]]:
    root = Path(repo_root)
    rows = []
    for name, raw_path in config.input_paths.items():
        path = Path(str(raw_path))
        resolved = path if path.is_absolute() else root / path
        rows.append(
            {
                "name": name,
                "path": str(path),
                "exists": resolved.exists(),
                "sha256": sha256_file(resolved) if resolved.is_file() else None,
            }
        )
    return rows


def to_pretty_json(data: Mapping[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)


def _git(args: list[str], repo_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(repo_root),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()
