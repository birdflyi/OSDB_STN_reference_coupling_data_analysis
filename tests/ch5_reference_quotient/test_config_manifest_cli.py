from pathlib import Path
import re

import GH_CoRE

from script.ch5_reference_quotient.cli import build_parser, main
from script.ch5_reference_quotient.config import load_config, validate_config
from script.ch5_reference_quotient.manifest import runtime_versions


CONFIG_PATH = "configs/ch5_reference_quotient_p0.yaml"


def test_config_validates_with_read_only_inputs():
    config = load_config(CONFIG_PATH)
    assert validate_config(config, Path.cwd()) == []
    assert config.run_id_prefix == "ch5_refq_p0"
    assert "reference_quotient_p0_frozen" in str(config.output_root)


def test_cli_dry_run_uses_reference_quotient_names():
    exit_code = main(["--config", CONFIG_PATH, "--validate-config", "--dry-run"])
    assert exit_code == 0
    help_text = build_parser().format_help()
    assert "Reference Quotient" in help_text
    assert "ch5_reference_quotient_p0.yaml" in help_text
    assert "direct_reference_coupling" not in help_text


def test_runtime_manifest_records_gh_core_version():
    assert runtime_versions()["gh_core"] == GH_CoRE.__version__


def test_p0_runtime_lock_matches_current_environment():
    lock_text = Path("environment/p0-requirements-lock.txt").read_text(encoding="utf-8")
    expected = {
        "python": "3.9.13",
        "numpy": "1.26.4",
        "pandas": "1.4.4",
        "scipy": "1.13.1",
        "networkx": "3.1",
        "gh-core": "2.3.1",
    }
    assert f"Python: {expected['python']}" in lock_text
    for package, version in expected.items():
        if package == "python":
            continue
        assert re.search(rf"^{re.escape(package)}=={re.escape(version)}$", lock_text, re.MULTILINE)
    runtime = runtime_versions()
    assert runtime["python"] == expected["python"]
    assert runtime["numpy"] == expected["numpy"]
    assert runtime["pandas"] == expected["pandas"]
    assert runtime["scipy"] == expected["scipy"]
    assert runtime["networkx"] == expected["networkx"]
    assert runtime["gh_core"] == expected["gh-core"]
