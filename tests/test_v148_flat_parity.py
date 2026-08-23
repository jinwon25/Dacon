from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from src.champion.v148_flat_build_package import (
    DROPPED,
    MODEL_MAP,
    ORIGINAL_SHA256,
    transform_base_ensemble,
    transform_strict_asof,
    transform_strict_overlay,
)


ARTIFACT_DIR = Path("artifacts/v148_flat_20260823_01")
FLAT_ZIP = ARTIFACT_DIR / "submit_v148_flat.zip"
MANIFEST = ARTIFACT_DIR / "manifest.json"
ORIGINAL_ZIP = Path(
    "artifacts/v148_v142_v138_blend_package_20260823_01/submit_v148.zip"
)
PARITY_TOLERANCE = 1e-12

pytestmark = pytest.mark.skipif(
    not (FLAT_ZIP.exists() and MANIFEST.exists()),
    reason="flat v148 package has not been built in this checkout",
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_recorded_parity_is_within_tolerance(manifest: dict) -> None:
    parity = manifest["parity"]
    assert parity["parity_passed"] is True
    assert parity["max_abs_diff"] <= PARITY_TOLERANCE
    assert parity["rows"] >= 100_000
    # the residual must be no larger than the original package's own
    # run-to-run floating-point nondeterminism
    assert parity["max_abs_diff"] <= parity["original_vs_original_max_abs"]


def test_parity_covers_every_domain_and_cold_start(manifest: dict) -> None:
    parity = manifest["parity"]
    for domain in ("R_CORE", "R_ANCHOR", "F"):
        assert parity["per_domain"][domain]["rows"] > 0
        assert parity["per_domain"][domain]["max_abs_diff"] <= PARITY_TOLERANCE
    assert parity["cold_start_rows"] > 0
    assert parity["cold_start_max_abs_diff"] <= PARITY_TOLERANCE
    assert len(parity["source_seasons_sampled"]) >= 5
    assert len(parity["months"]) >= 6


def test_row_independence_invariants_hold(manifest: dict) -> None:
    parity = manifest["parity"]
    assert parity["shuffle_max_abs"] <= PARITY_TOLERANCE
    assert parity["partition_max_abs"] <= PARITY_TOLERANCE


def test_flat_runtime_is_not_slower_than_the_original(manifest: dict) -> None:
    benchmark = manifest["runtime_benchmark"]
    assert benchmark["rows"] == 245_789
    assert benchmark["max_abs_diff"] <= PARITY_TOLERANCE
    # measured back to back in one session, so the ratio is load-independent
    assert benchmark["flat_vs_original_ratio"] <= 1.05


def test_every_model_artifact_is_byte_identical(manifest: dict) -> None:
    identity = manifest["artifact_byte_identity"]
    assert identity["mismatches"] == []
    assert identity["verified"] == 78


def test_flat_layout_is_single_generation() -> None:
    with zipfile.ZipFile(FLAT_ZIP) as archive:
        names = archive.namelist()
    assert "script.py" in names
    assert "requirements.txt" in names
    assert "lib/paths.py" in names
    # no genealogy nesting survives
    assert not any("parent/parent" in name for name in names)
    assert not any(name.startswith("model/v124/") for name in names)
    assert max(name.count("/") for name in names) <= 3
    # the flat entry point resolves components statically
    source = zipfile.ZipFile(FLAT_ZIP).read("script.py").decode("utf-8")
    assert "import importlib" not in source
    assert "from lib import" in source


def test_no_component_retains_a_second_entry_point() -> None:
    with zipfile.ZipFile(FLAT_ZIP) as archive:
        for name in archive.namelist():
            if not name.startswith("lib/") or not name.endswith(".py"):
                continue
            source = archive.read(name).decode("utf-8")
            assert "__main__" not in source, name
            assert "import importlib" not in source, name


def test_original_package_is_untouched() -> None:
    from src.champion.v148_flat_build_package import _sha256_file

    assert _sha256_file(ORIGINAL_ZIP) == ORIGINAL_SHA256


def test_dropped_members_are_only_entry_points_and_duplicate_requirements() -> None:
    assert set(DROPPED) == {
        "script.py",
        "requirements.txt",
        "model/h1/script.py",
        "model/h1/requirements.txt",
        "model/v124/script.py",
        "model/v124/requirements.txt",
        "model/v124/model/components/parent_script.py",
        "model/v124/model/components/v104_features.py",
        "model/v124/model/parent/components/parent_script.py",
        "model/v124/model/parent/parent/components/champion_script.py",
        "model/v124/model/parent/parent/components/strict_script.py",
    }
    assert len(MODEL_MAP) == 7


def test_transformations_fail_loudly_when_anchors_move() -> None:
    for transform in (
        transform_base_ensemble,
        transform_strict_asof,
        transform_strict_overlay,
    ):
        with pytest.raises(ValueError):
            transform("def unrelated():\n    return 0\n")


# Names describe function or domain, never lineage ("v124"), status
# ("champion", "legacy"), or a leaderboard goal ("target1160").  Domain codes
# R/R_CORE/R_ANCHOR/F and hyper-parameter encodings are not lineage.
LINEAGE_TOKENS = ("v82", "v84", "v104", "v124", "champion", "legacy", "target1160")


def test_module_and_directory_names_carry_no_lineage_or_status() -> None:
    """Weight *filenames* are exempt: see the module docstring in the builder."""
    with zipfile.ZipFile(FLAT_ZIP) as archive:
        names = archive.namelist()
    checked = [
        name
        for name in names
        if name.startswith("lib/") or "/" not in name
    ]
    checked += sorted({f"model/{name.split('/')[1]}/" for name in names if name.startswith("model/") and "/" in name[6:]})
    offenders = [
        name for name in checked if any(bad in name.lower() for bad in LINEAGE_TOKENS)
    ]
    assert offenders == []


def test_every_component_module_is_present() -> None:
    with zipfile.ZipFile(FLAT_ZIP) as archive:
        names = set(archive.namelist())
    for module in (
        "base_ensemble",
        "strict_asof",
        "strict_overlay",
        "futures_fm_overlay",
        "conditional_overlay",
        "row_local_features",
        "form_context_rf",
    ):
        assert f"lib/{module}.py" in names
    for directory in ("base_ensemble", "strict_asof", "futures_fm", "regular_fm", "form_context_rf"):
        assert any(name.startswith(f"model/{directory}/") for name in names), directory


def test_architecture_document_describes_the_pipeline() -> None:
    with zipfile.ZipFile(FLAT_ZIP) as archive:
        assert "ARCHITECTURE.md" in archive.namelist()
        document = archive.read("ARCHITECTURE.md").decode("utf-8")
    for module in ("base_ensemble", "strict_overlay", "conditional_overlay"):
        assert module in document
    # the blend the package actually applies
    assert "0.85" in document and "0.15" in document and "0.5" in document
