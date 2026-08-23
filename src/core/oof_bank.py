"""Frozen OOF candidate bank loading and rebasing.

Moved verbatim from ``src/v80_oof_covariance_stack.py`` during the core extraction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


EPS = 0.001


AXIS_FILES = {
    "late_2023": "selection_late_2023.npz",
    "full_2024": "outer_full_2024.npz",
    "late_2024": "replication_late_2024.npz",
}


CHAMPION_FILES = {
    "late_2023": "y2023_early_to_late.npz",
    "full_2024": "y2023_to_y2024.npz",
    "late_2024": "y2024_early_to_late.npz",
}


def rebase_frozen_candidate(
    incumbent: np.ndarray,
    saved_parent: np.ndarray,
    saved_candidate: np.ndarray,
) -> np.ndarray:
    """Preserve a frozen probability shift above the exact current parent."""

    current = np.asarray(incumbent, dtype=np.float64)
    old = np.asarray(saved_parent, dtype=np.float64)
    candidate = np.asarray(saved_candidate, dtype=np.float64)
    if not (current.ndim == 1 and current.shape == old.shape == candidate.shape):
        raise ValueError("rebase arrays must be aligned one-dimensional vectors")
    if not (np.isfinite(current).all() and np.isfinite(old).all() and np.isfinite(candidate).all()):
        raise ValueError("rebase arrays contain non-finite values")
    return np.clip(current + candidate - old, EPS, 1.0 - EPS)


def _archive(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as saved:
        return {name: saved[name] for name in saved.files}


def _assert_target(expected: np.ndarray, archive: dict[str, np.ndarray], label: str) -> None:
    target = np.asarray(archive["target"], dtype=np.float64)
    if not np.array_equal(np.asarray(expected, dtype=np.float64), target):
        raise ValueError(f"target/order mismatch: {label}")


def _load_parent_axis(final_parent_dir: Path, axis: str) -> dict[str, np.ndarray]:
    saved = _archive(final_parent_dir / AXIS_FILES[axis])
    required = {"target", "final_gate_parent", "domain3", "game_month"}
    missing = sorted(required - set(saved))
    if missing:
        raise ValueError(f"final parent axis {axis} missing {missing}")
    return {
        "target": np.asarray(saved["target"], dtype=np.float64),
        "parent": np.asarray(saved["final_gate_parent"], dtype=np.float64),
        "domain3": np.asarray(saved["domain3"]).astype(str),
        "month": np.asarray(saved["game_month"], dtype=np.int16),
    }


def _load_common_candidates(
    project: Path,
    parent_axis: dict[str, np.ndarray],
    axis: str,
) -> dict[str, np.ndarray]:
    artifacts = project / "artifacts"
    target = parent_axis["target"]
    parent = parent_axis["parent"]
    champion = _archive(
        artifacts / "champion_oof_20260817_01" / CHAMPION_FILES[axis]
    )
    _assert_target(target, champion, f"champion/{axis}")
    candidates = {
        f"base_{name}": np.asarray(champion[name], dtype=np.float64)
        for name in ("v17", "v19", "v20", "v21")
    }

    v26 = _archive(
        artifacts / "v26_exact_diversity_20260817_01" / AXIS_FILES[axis]
    )
    _assert_target(target, v26, f"v26/{axis}")
    candidates["v26_exact_diversity_shift"] = rebase_frozen_candidate(
        parent, v26["v25"], v26["candidate"]
    )

    v49 = _archive(
        artifacts / "v49_temporal_convex_stack_20260817_01" / AXIS_FILES[axis]
    )
    _assert_target(target, v49, f"v49/{axis}")
    candidates["v49_temporal_stack_shift"] = rebase_frozen_candidate(
        parent, v49["v27"], v49["candidate"]
    )
    for name, value in candidates.items():
        if value.shape != target.shape or not np.isfinite(value).all():
            raise ValueError(f"invalid common candidate {name}/{axis}")
    return candidates


def _load_full24_diagnostic_candidates(
    project: Path,
    parent_axis: dict[str, np.ndarray],
    common: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    artifacts = project / "artifacts"
    target = parent_axis["target"]
    parent = parent_axis["parent"]
    output = dict(common)
    registries = {
        "v46_sparse_logit_shift": ("v46_sparse_logit_20260817_01", "v27", "candidate"),
        "v47_pitcher_role_shift": ("v47_pitcher_role_20260817_01", "v27", "candidate"),
        "v48_level_transition_shift": ("v48_level_transition_20260817_01", "v27", "candidate"),
        "v50_low_rank_context_shift": ("v50_low_rank_pitcher_context_20260817_01", "v27", "candidate"),
        "v53_factorization_shift": ("v53_factorization_offset_20260817_01", "v27", "candidate"),
        "v56_shared_horizon_fm_shift": ("v56_shared_horizon_fm_20260817_01", "v27", "candidate"),
        "v57_independent_blend_shift": ("v57_public_strict_blend_20260817_01", "v27", "candidate"),
    }
    for name, (directory, parent_key, candidate_key) in registries.items():
        saved = _archive(artifacts / directory / "outer_full_2024.npz")
        _assert_target(target, saved, name)
        output[name] = rebase_frozen_candidate(
            parent, saved[parent_key], saved[candidate_key]
        )

    tabm = _archive(
        artifacts / "v45_tabm_mini_20260817_01" / "audit_predictions.npz"
    )
    if not np.array_equal(target, np.asarray(tabm["full_target"], dtype=np.float64)):
        raise ValueError("target/order mismatch: v45/full_2024")
    output["v45_tabm_shift"] = rebase_frozen_candidate(
        parent, tabm["full_parent"], tabm["full_candidate"]
    )
    return output
