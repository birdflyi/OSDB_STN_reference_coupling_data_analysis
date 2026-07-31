from datetime import datetime, timezone

import pytest

from script.ch5_direct_reference_coupling.cli import main
from script.ch5_direct_reference_coupling.config import load_config, summarize_inputs, validate_config
from script.ch5_direct_reference_coupling.manifest import build_run_id, eligible_for_freeze
from script.ch5_direct_reference_coupling.result_validation import assert_not_eligible_for_freeze
from script.ch5_direct_reference_coupling.rq3_subdomain_comparison import filter_category_mode


CONFIG_PATH = "configs/ch5_direct_reference_coupling_p0.yaml"


def test_config_validates_and_lists_inputs():
    config = load_config(CONFIG_PATH)

    assert validate_config(config) == []
    inputs = summarize_inputs(config)
    assert any(row["name"] == "dbms_repos_key_features" for row in inputs)


def test_run_id_contains_prefix_commit_and_prep_marker():
    config = load_config(CONFIG_PATH)
    run_id = build_run_id(config, "abc1234", datetime(2026, 7, 31, tzinfo=timezone.utc))

    assert run_id == "ch5_drc_20260731T000000Z_abc1234_prep"


def test_preparation_freeze_guard():
    assert eligible_for_freeze(True, True, True, True, True) is True
    with pytest.raises(AssertionError):
        assert_not_eligible_for_freeze(True)


def test_cli_dry_run_does_not_execute_full_run(capsys):
    exit_code = main(
        [
            "--config",
            CONFIG_PATH,
            "--validate-config",
            "--dry-run",
            "--show-inputs",
            "--show-planned-outputs",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Config validation passed." in captured.out
    assert "eligible_for_freeze" in captured.out
    assert "direct_reference_edges.csv" in captured.out


def test_mixed_category_strict_mode_excludes_multilabels():
    import pandas as pd

    records = pd.DataFrame({"category_label": ["Relational", "Graph;RDF", "Vector|Search"]})
    strict = filter_category_mode(records, mode="exclude_mixed_or_multilabel")

    assert strict["category_label"].tolist() == ["Relational"]
