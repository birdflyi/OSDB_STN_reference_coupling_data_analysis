"""Dry-run CLI for the Chapter 5 direct reference coupling preparation scaffold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from .config import as_plain_dict, load_config, summarize_inputs, validate_config
from .manifest import build_run_id, build_run_manifest, git_info, to_pretty_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chapter 5 Direct Reference Coupling runner")
    parser.add_argument("--config", required=True, help="Path to the Ch5 pipeline config")
    parser.add_argument("--dry-run", action="store_true", help="Plan the run without writing outputs")
    parser.add_argument("--validate-config", action="store_true", help="Validate config and exit")
    parser.add_argument("--show-inputs", action="store_true", help="Print configured input paths")
    parser.add_argument("--show-planned-outputs", action="store_true", help="Print planned staging output paths")
    parser.add_argument("--execute", action="store_true", help="Reserved for explicit future full execution")
    parser.add_argument("--run-stage", choices=["p0"], help="Reserved explicit stage selector for future execution")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
    errors = validate_config(config)
    repo_root = Path.cwd()
    info = git_info(repo_root)
    run_id = build_run_id(config, info.get("short_commit", "unknown"))

    if errors:
        print("Config validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    if args.validate_config:
        print("Config validation passed.")

    if args.show_inputs:
        print(to_pretty_json({"inputs": summarize_inputs(config, repo_root)}))

    if args.show_planned_outputs:
        print(to_pretty_json({"planned_outputs": config.planned_outputs(run_id)}))

    if args.dry_run:
        manifest = build_run_manifest(
            config=config,
            run_id=run_id,
            repo_root=repo_root,
            run_status="dry_run",
            validation_passed=True,
            git_commit_pushed=False,
        )
        print(to_pretty_json({"dry_run_manifest": manifest}))
        return 0

    if args.execute or args.run_stage:
        print(
            "Full P0 execution is intentionally not implemented in the preparation scaffold. "
            "Use dry-run until the P0 recalculation stage is approved.",
            file=sys.stderr,
        )
        return 2

    print("No execution requested. Use --dry-run, --validate-config, --show-inputs, or --show-planned-outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
