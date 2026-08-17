"""Configuration for the Chapter 5 Reference Quotient frozen run."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


REQUIRED_KEYS = {
    "study_year",
    "data_version",
    "analysis_seed_activity_threshold",
    "expected_candidate_seed_count",
    "expected_analysis_seed_count",
    "relation_type",
    "reference_dedup_rule",
    "membership_prefix",
    "membership_rule",
    "quotient_self_loop_policy",
    "rq2_cross_project_self_loop_policy",
    "edge_aggregation_rule",
    "category_assignment_mode",
    "random_seed",
    "source_repository",
    "input_paths",
    "frozen_output_root",
}


@dataclass(frozen=True)
class RefQConfig:
    raw: Mapping[str, Any]
    path: Optional[Path] = None

    def get_int(self, key: str, default: int = 0) -> int:
        return int(self.raw.get(key, default))

    @property
    def input_paths(self) -> Mapping[str, str]:
        value = self.raw.get("input_paths", {})
        return value if isinstance(value, Mapping) else {}

    @property
    def source_repository(self) -> Mapping[str, str]:
        value = self.raw.get("source_repository", {})
        return value if isinstance(value, Mapping) else {}

    @property
    def output_root(self) -> Path:
        return Path(str(self.raw["frozen_output_root"]))

    @property
    def run_id_prefix(self) -> str:
        return str(self.raw.get("run_id_prefix", "ch5_refq_p0"))


def load_config(path: str | Path) -> RefQConfig:
    config_path = Path(path)
    data = _load_yaml_like(config_path.read_text(encoding="utf-8"))
    return RefQConfig(raw=data, path=config_path)


def validate_config(config: RefQConfig, workspace_root: str | Path = ".") -> List[str]:
    errors: List[str] = []
    missing = sorted(REQUIRED_KEYS - set(config.raw))
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")
    if config.raw.get("relation_type") != "Reference":
        errors.append("relation_type must be Reference")
    if config.raw.get("reference_dedup_rule") != "none":
        errors.append("the frozen P0 run currently requires reference_dedup_rule=none")
    if config.raw.get("quotient_self_loop_policy") != "preserve":
        errors.append("quotient_self_loop_policy must be preserve")
    if config.raw.get("rq2_cross_project_self_loop_policy") != "exclude":
        errors.append("rq2_cross_project_self_loop_policy must be exclude")
    if config.raw.get("category_assignment_mode") not in {"include_mixed", "exclude_mixed_or_multilabel"}:
        errors.append("invalid category_assignment_mode")
    if not str(config.raw.get("data_version", "")).strip():
        errors.append("data_version must be non-empty")
    for key in ("expected_candidate_seed_count", "expected_analysis_seed_count", "csv_chunk_size"):
        try:
            if int(config.raw.get(key, 0)) <= 0:
                errors.append(f"{key} must be positive")
        except (TypeError, ValueError):
            errors.append(f"{key} must be an integer")
    for name, raw_path in config.input_paths.items():
        if not Path(str(raw_path)).exists():
            errors.append(f"missing input path {name}: {raw_path}")
    source_path = Path(str(config.source_repository.get("path", "")))
    if not source_path.is_dir():
        errors.append(f"missing source repository: {source_path}")
    output = config.output_root
    if output.is_absolute():
        errors.append("frozen_output_root must be relative to the writable workspace")
    else:
        resolved = (Path(workspace_root) / output).resolve()
        root = Path(workspace_root).resolve()
        if root not in resolved.parents:
            errors.append("frozen_output_root escapes the writable workspace")
    return errors


def resolved_inputs(config: RefQConfig) -> Dict[str, Path]:
    return {name: Path(str(value)).resolve() for name, value in config.input_paths.items()}


def _load_yaml_like(text: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return _parse_simple_yaml(text)
    loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError("config root must be a mapping")
    return loaded


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    maps: list[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, value = line.strip().split(":", 1)
        while maps[-1][0] >= indent:
            maps.pop()
        target = maps[-1][1]
        value = value.strip()
        if not value:
            nested: Dict[str, Any] = {}
            target[key.strip()] = nested
            maps.append((indent, nested))
        else:
            target[key.strip()] = _parse_scalar(value)
    return root


def _strip_comment(line: str) -> str:
    in_single = in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:index]
    return line


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") or value.startswith("{"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)
    for converter in (int, float):
        try:
            return converter(value)
        except ValueError:
            pass
    return value.strip('"').strip("'")
