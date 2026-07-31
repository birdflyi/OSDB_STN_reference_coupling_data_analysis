"""Configuration helpers for the Chapter 5 direct reference coupling pipeline."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


REQUIRED_KEYS = {
    "study_year",
    "candidate_seed_source",
    "analysis_seed_activity_threshold",
    "event_types",
    "relation_type",
    "reference_dedup_rule",
    "source_granularity",
    "target_granularity",
    "include_self_reference_in_behavior_view",
    "drop_self_loop_in_cross_project_network",
    "repo_nodes_only",
    "external_service_filter",
    "node_weight_threshold",
    "edge_weight_threshold",
    "category_assignment_mode",
    "random_seed",
    "input_paths",
    "staging_output_root",
}

DEDUP_RULES = {"none", "event_source_target"}
CATEGORY_MODES = {"include_mixed", "exclude_mixed_or_multilabel"}


@dataclass(frozen=True)
class Ch5Config:
    """Thin wrapper around a resolved config mapping."""

    raw: Mapping[str, Any]
    path: Optional[Path] = None

    @property
    def study_year(self) -> int:
        return int(self.raw["study_year"])

    @property
    def random_seed(self) -> int:
        return int(self.raw["random_seed"])

    @property
    def input_paths(self) -> Mapping[str, str]:
        value = self.raw.get("input_paths", {})
        return value if isinstance(value, Mapping) else {}

    @property
    def staging_output_root(self) -> Path:
        return Path(str(self.raw["staging_output_root"]))

    @property
    def run_id_prefix(self) -> str:
        return str(self.raw.get("run_id_prefix", "ch5_drc"))

    def planned_outputs(self, run_id: str) -> Dict[str, str]:
        root = self.staging_output_root / run_id
        return {
            "run_manifest": str(root / "run_manifest.json"),
            "resolved_config": str(root / "resolved_config.yaml"),
            "input_manifest": str(root / "input_manifest.csv"),
            "output_manifest": str(root / "output_manifest.csv"),
            "run_log": str(root / "run.log"),
            "validation_report": str(root / "validation_report.json"),
            "directed_edges": str(root / "direct_reference_edges.csv"),
            "rq2a_source_view": str(root / "rq2a_source_view.csv"),
            "rq2b_target_view": str(root / "rq2b_target_view.csv"),
            "rq2c_undirected_edges": str(root / "rq2c_undirected_edges.csv"),
        }


def load_config(path: str | Path) -> Ch5Config:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    data = _load_yaml_like(text)
    return Ch5Config(raw=data, path=config_path)


def validate_config(config: Ch5Config) -> List[str]:
    errors: List[str] = []
    missing = sorted(REQUIRED_KEYS - set(config.raw))
    if missing:
        errors.append(f"missing required keys: {', '.join(missing)}")

    if config.raw.get("relation_type") != "Reference":
        errors.append("relation_type must be Reference for Direct Reference Coupling")

    if config.raw.get("reference_dedup_rule") not in DEDUP_RULES:
        errors.append(f"reference_dedup_rule must be one of {sorted(DEDUP_RULES)}")

    if config.raw.get("category_assignment_mode") not in CATEGORY_MODES:
        errors.append(f"category_assignment_mode must be one of {sorted(CATEGORY_MODES)}")

    for key in ("analysis_seed_activity_threshold", "node_weight_threshold", "edge_weight_threshold"):
        try:
            if int(config.raw.get(key, -1)) < 0:
                errors.append(f"{key} must be non-negative")
        except (TypeError, ValueError):
            errors.append(f"{key} must be an integer")

    if bool(config.raw.get("execute_by_default", False)):
        errors.append("execute_by_default must not be true in the preparation scaffold")

    if not isinstance(config.raw.get("event_types"), list) or not config.raw.get("event_types"):
        errors.append("event_types must be a non-empty list")

    if not isinstance(config.raw.get("external_service_filter"), list):
        errors.append("external_service_filter must be a list")

    if not isinstance(config.raw.get("input_paths"), Mapping):
        errors.append("input_paths must be a mapping")

    return errors


def summarize_inputs(config: Ch5Config, repo_root: str | Path = ".") -> List[Dict[str, Any]]:
    root = Path(repo_root)
    rows: List[Dict[str, Any]] = []
    for name, raw_path in config.input_paths.items():
        path = Path(str(raw_path))
        resolved = path if path.is_absolute() else root / path
        rows.append(
            {
                "name": name,
                "path": str(path),
                "exists": resolved.exists(),
                "kind": "dir" if resolved.is_dir() else "file" if resolved.is_file() else "missing",
            }
        )
    return rows


def _load_yaml_like(text: str) -> Dict[str, Any]:
    """Load YAML with PyYAML when available, otherwise parse the small subset we use."""

    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None

    if yaml is not None:
        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError("config root must be a mapping")
        return loaded

    return _parse_simple_yaml(text)


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    current_map: Optional[Dict[str, Any]] = None
    current_indent = 0

    for raw_line in text.splitlines():
        line = _strip_comment(raw_line).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"unsupported config line: {raw_line}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value == "":
            nested: Dict[str, Any] = {}
            root[key] = nested
            current_map = nested
            current_indent = indent
            continue

        parsed = _parse_scalar(value)
        if current_map is not None and indent > current_indent:
            current_map[key] = parsed
        else:
            root[key] = parsed
            current_map = None
            current_indent = 0

    return root


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
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
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip('"').strip("'")


def require_valid_config(config: Ch5Config) -> None:
    errors = validate_config(config)
    if errors:
        raise ValueError("; ".join(errors))


def as_plain_dict(config: Ch5Config) -> Dict[str, Any]:
    return dict(config.raw)
