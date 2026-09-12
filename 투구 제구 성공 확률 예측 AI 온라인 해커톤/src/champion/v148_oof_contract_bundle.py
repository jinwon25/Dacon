"""Build the v148 champion OOF evidence bundle in the team contract format.

The v147 blend audit stored the v142/v138/v124 axes for full-2024 but selected
weight ``0.05``.  The deployed champion ``submit_v148.zip`` uses
``blend_weight_toward_v138 = 0.15``, so the champion's own OOF is reconstructed
here from the stored parent axes rather than re-run.

Alignment and reconstruction are both verified before anything is written:

* the audit arrays are positionally aligned with the frozen contract axis
  (``active_mask`` must equal ``domain3 == "R_CORE"``, and ``late_2024`` must
  equal ``full_2024`` restricted to ``game_month >= 8``);
* the reconstruction formula must reproduce the gain figures v147 published for
  weight ``0.05`` exactly.

Both are hard failures.  A misaligned bundle would silently corrupt every
residual-correlation study built on it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.core.axis_metrics import _axis_metrics


PROTOCOL = "V148_OOF_CONTRACT_BUNDLE_V1"

DEPLOYED_WEIGHT = 0.15
AUDIT_WEIGHT = 0.05
CLIP_LOW = 0.001
CLIP_HIGH = 0.999
LATE_MONTH = 8

# Published in artifacts/v147_v142_v138_blend_audit_20260823_01/summary.json.
V147_GAIN = 7.198934146857027
V147_LATE_GAIN = 12.964273655657848

METADATA_KEYS = (
    "raw_index",
    "target",
    "exact_mask",
    "season",
    "game_month",
    "domain3",
    "pitcher_id",
    "batter_id",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def blend(v142: np.ndarray, v138: np.ndarray, weight: float) -> np.ndarray:
    """Champion bridge: move ``weight`` of the way from v142 toward v138."""
    return np.clip(v142 + weight * (v138 - v142), CLIP_LOW, CLIP_HIGH)


def verify_alignment(
    axis: dict[str, np.ndarray],
    audit: dict[str, np.ndarray],
    absolute: dict[str, np.ndarray],
) -> dict[str, object]:
    """Prove the audit arrays share the contract axis row order."""
    rows = len(axis["raw_index"])
    if len(audit["full_2024"]) != rows:
        raise ValueError("audit full_2024 length does not match the contract axis")

    seasons = np.unique(axis["season"])
    if seasons.tolist() != [2024]:
        raise ValueError(f"contract axis is not full-2024: {seasons.tolist()}")
    if not np.all(np.diff(axis["raw_index"]) > 0):
        raise ValueError("contract raw_index is not strictly increasing")

    r_core = axis["domain3"].astype(str) == "R_CORE"
    if not np.array_equal(absolute["active_mask"].astype(bool), r_core):
        raise ValueError("active_mask does not match contract R_CORE rows")

    late = axis["game_month"].astype(np.int16) >= LATE_MONTH
    if len(audit["late_2024"]) != int(late.sum()):
        raise ValueError("late_2024 length does not match contract game_month >= 8")
    if not np.array_equal(audit["late_2024"], audit["full_2024"][late]):
        raise ValueError("late_2024 is not full_2024 restricted to late months")

    return {
        "rows": int(rows),
        "r_core_rows": int(r_core.sum()),
        "late_rows": int(late.sum()),
        "active_mask_equals_r_core": True,
        "late_axis_consistent": True,
        "raw_index_strictly_increasing": True,
    }


def verify_reconstruction(
    axis: dict[str, np.ndarray], audit: dict[str, np.ndarray]
) -> dict[str, object]:
    """Reproduce v147's published weight-0.05 figures with the same formula."""
    v124 = audit["v124_full_2024"]
    candidate = blend(audit["v142_full_2024"], audit["v138_full_2024"], AUDIT_WEIGHT)

    if not np.array_equal(candidate, audit["full_2024"]):
        raise ValueError("stored full_2024 is not the weight-0.05 blend")

    gain = float(_axis_metrics({**axis, "parent": v124}, candidate)["gain"])
    late = axis["game_month"].astype(np.int16) >= LATE_MONTH
    late_axis = {
        key: value[late] if getattr(value, "shape", None) == axis["target"].shape else value
        for key, value in axis.items()
    }
    late_gain = float(
        _axis_metrics({**late_axis, "parent": v124[late]}, candidate[late])["gain"]
    )

    if abs(gain - V147_GAIN) > 1e-9 or abs(late_gain - V147_LATE_GAIN) > 1e-9:
        raise ValueError(
            f"reconstruction does not reproduce v147: {gain} / {late_gain}"
        )
    return {
        "audit_weight": AUDIT_WEIGHT,
        "reproduced_gain_vs_v124": gain,
        "published_gain_vs_v124": V147_GAIN,
        "reproduced_late_gain_vs_v124": late_gain,
        "published_late_gain_vs_v124": V147_LATE_GAIN,
        "stored_axis_matches_formula": True,
    }


def run(contract_axis: Path, audit_dir: Path, absolute_dir: Path, output_dir: Path) -> dict:
    with np.load(contract_axis, allow_pickle=False) as saved:
        axis = {key: saved[key] for key in saved.files}
    with np.load(audit_dir / "selected_axes.npz", allow_pickle=False) as saved:
        audit = {key: saved[key] for key in saved.files}
    with np.load(absolute_dir / "selected_axes.npz", allow_pickle=False) as saved:
        absolute = {key: saved[key] for key in saved.files}

    alignment = verify_alignment(axis, audit, absolute)
    reconstruction = verify_reconstruction(axis, audit)

    champion = blend(audit["v142_full_2024"], audit["v138_full_2024"], DEPLOYED_WEIGHT)
    if not np.isfinite(champion).all():
        raise ValueError("champion OOF contains non-finite values")
    if champion.min() < CLIP_LOW or champion.max() > CLIP_HIGH:
        raise ValueError("champion OOF escaped the clip range")

    v124 = audit["v124_full_2024"]
    late = axis["game_month"].astype(np.int16) >= LATE_MONTH
    late_axis = {
        key: value[late] if getattr(value, "shape", None) == axis["target"].shape else value
        for key, value in axis.items()
    }
    champion_gain = float(_axis_metrics({**axis, "parent": v124}, champion)["gain"])
    champion_late_gain = float(
        _axis_metrics({**late_axis, "parent": v124[late]}, champion[late])["gain"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {key: axis[key] for key in METADATA_KEYS}
    payload["parent"] = champion
    payload["v142_full_2024"] = audit["v142_full_2024"]
    payload["v138_full_2024"] = audit["v138_full_2024"]
    payload["v124_full_2024"] = v124
    bundle_path = output_dir / "v148_full_2024.npz"
    np.savez_compressed(bundle_path, **payload)

    result = {
        "protocol": PROTOCOL,
        "champion": "submit_v148.zip",
        "champion_public_score": 1170.3014697177,
        "deployed_weight_toward_v138": DEPLOYED_WEIGHT,
        "formula": "clip(v142 + 0.15 * (v138 - v142), 0.001, 0.999)",
        "alignment_verification": alignment,
        "reconstruction_verification": reconstruction,
        "champion_axis": {
            "rows": int(len(champion)),
            "mean": float(champion.mean()),
            "min": float(champion.min()),
            "max": float(champion.max()),
            "gain_vs_v124": champion_gain,
            "late_gain_vs_v124": champion_late_gain,
        },
        "files": {
            bundle_path.name: {
                "sha256": _sha256(bundle_path),
                "bytes": bundle_path.stat().st_size,
            }
        },
        "restrictions": {
            "contains_official_train_labels": True,
            "contains_player_identifiers": True,
            "official_team_members_only": True,
            "public_redistribution_prohibited": True,
            "third_party_artifacts_excluded": True,
            "full_2024_is_development_contaminated": True,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-axis", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--absolute-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.contract_axis, args.audit_dir, args.absolute_dir, args.output_dir)


if __name__ == "__main__":
    main()
