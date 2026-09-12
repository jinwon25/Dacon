"""Pin the v148 OOF bundle formula, schema, and verification gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.champion.v148_oof_contract_bundle import (
    DEPLOYED_WEIGHT,
    METADATA_KEYS,
    blend,
    verify_alignment,
    verify_reconstruction,
)


BUNDLE = Path("artifacts/oof_champion_1170")
CONTRACT = Path(
    "artifacts/v94_multi_origin_champion_contract_20260822_01/v84_full_2024.npz"
)
AUDIT = Path("artifacts/v147_v142_v138_blend_audit_20260823_01/selected_axes.npz")
ABSOLUTE = Path("artifacts/v141_v124_absolute_h1_c3_20260823_01/selected_axes.npz")


def test_deployed_weight_matches_champion_config():
    config = json.loads(
        Path("research/configs/v148_v142_v138_blend_package.json").read_text(encoding="utf-8")
    )
    assert float(config["blend_weight_toward_v138"]) == DEPLOYED_WEIGHT


def test_blend_endpoints_and_clip():
    v142 = np.array([0.2, 0.9, 0.0005], dtype=np.float64)
    v138 = np.array([0.6, 0.1, 0.0005], dtype=np.float64)
    assert np.allclose(blend(v142, v138, 0.0), np.clip(v142, 0.001, 0.999))
    assert np.allclose(blend(v142, v138, 1.0), np.clip(v138, 0.001, 0.999))
    mid = blend(v142, v138, DEPLOYED_WEIGHT)
    assert mid.min() >= 0.001 and mid.max() <= 0.999


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


@pytest.mark.skipif(not CONTRACT.exists() or not AUDIT.exists(), reason="artifacts absent")
def test_alignment_and_reconstruction_gates_pass():
    axis = _load(CONTRACT)
    audit = _load(AUDIT)
    absolute = _load(ABSOLUTE)

    alignment = verify_alignment(axis, audit, absolute)
    assert alignment["rows"] == 253507
    assert alignment["r_core_rows"] == 178729
    assert alignment["late_rows"] == 76896

    reconstruction = verify_reconstruction(axis, audit)
    assert reconstruction["stored_axis_matches_formula"] is True


@pytest.mark.skipif(not CONTRACT.exists() or not AUDIT.exists(), reason="artifacts absent")
def test_alignment_gate_rejects_shifted_mask():
    axis = _load(CONTRACT)
    audit = _load(AUDIT)
    absolute = _load(ABSOLUTE)
    broken = {**absolute, "active_mask": np.roll(absolute["active_mask"], 1)}
    with pytest.raises(ValueError, match="active_mask"):
        verify_alignment(axis, audit, broken)


@pytest.mark.skipif(not (BUNDLE / "v148_full_2024.npz").exists(), reason="bundle absent")
def test_bundle_schema_and_contents():
    bundle = _load(BUNDLE / "v148_full_2024.npz")
    for key in METADATA_KEYS:
        assert key in bundle
    for key in ("parent", "v142_full_2024", "v138_full_2024", "v124_full_2024"):
        assert key in bundle

    parent = bundle["parent"]
    assert len(parent) == 253507
    assert np.isfinite(parent).all()
    assert parent.min() >= 0.001 and parent.max() <= 0.999

    expected = blend(bundle["v142_full_2024"], bundle["v138_full_2024"], DEPLOYED_WEIGHT)
    assert np.array_equal(parent, expected)
    assert set(np.unique(bundle["domain3"]).astype(str)) == {"R_CORE", "R_ANCHOR", "F"}


@pytest.mark.skipif(not (BUNDLE / "manifest.json").exists(), reason="bundle absent")
def test_manifest_records_restrictions_and_verification():
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["deployed_weight_toward_v138"] == DEPLOYED_WEIGHT

    restrictions = manifest["restrictions"]
    assert restrictions["official_team_members_only"] is True
    assert restrictions["public_redistribution_prohibited"] is True
    assert restrictions["contains_official_train_labels"] is True
    assert restrictions["full_2024_is_development_contaminated"] is True

    reconstruction = manifest["reconstruction_verification"]
    assert (
        reconstruction["reproduced_gain_vs_v124"]
        == reconstruction["published_gain_vs_v124"]
    )
    assert (
        reconstruction["reproduced_late_gain_vs_v124"]
        == reconstruction["published_late_gain_vs_v124"]
    )


@pytest.mark.skipif(not (BUNDLE / "README.md").exists(), reason="bundle absent")
def test_readme_carries_the_disclosure_warning():
    text = (BUNDLE / "README.md").read_text(encoding="utf-8")
    assert "공개 금지" in text
    assert "development_contaminated" in text
