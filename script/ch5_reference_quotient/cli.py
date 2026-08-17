"""Command-line entry point for the Reference Quotient P0 run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

from .config import load_config, resolved_inputs, validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chapter 5 Reference Quotient P0 runner")
    parser.add_argument("--config", required=True, help="Path to ch5_reference_quotient_p0.yaml")
    parser.add_argument("--workspace-root", default=".", help="Writable code/output workspace")
    parser.add_argument("--validate-config", action="store_true", help="Validate configuration and inputs")
    parser.add_argument("--show-inputs", action="store_true", help="Print resolved read-only inputs")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing P0 outputs")
    parser.add_argument("--execute", action="store_true", help="Run the single P0 recalculation and freeze chain")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    workspace = Path(args.workspace_root).resolve()
    config = load_config(args.config)
    errors = validate_config(config, workspace)
    if errors:
        print("Config validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    if args.validate_config:
        print("Config validation passed.")
    if args.show_inputs:
        print(json.dumps({name: str(path) for name, path in resolved_inputs(config).items()}, indent=2))
    if args.execute:
        try:
            from .pipeline import RefQPipeline

            output = RefQPipeline(config, workspace).run()
        except Exception as exc:
            print(f"Reference Quotient P0 run failed: {exc}", file=sys.stderr)
            return 2
        print(f"Reference Quotient P0 frozen output: {output}")
        return 0
    if args.dry_run:
        print(f"Dry run passed; planned frozen output: {(workspace / config.output_root).resolve()}")
        return 0
    if not (args.validate_config or args.show_inputs):
        print("No action requested. Use --dry-run or --execute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
