"""Write a blended OOF + test score bank that the boundary code can consume directly.

Blend weights: 0.2 * old_attn_gru (selector_full) + 0.7 * new_attn_gru + 0.1 * new_tcn
(best from scripts/ensemble_old_attn_new_tcn.py greedy search).

Output:
  outputs/08_blend_selector/{oof,test}_selector_scores.npz
  with ens_scores filled from the blend and other keys passed through (for boundary
  candidate-name compatibility checks).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pipeline as P  # noqa: E402


WEIGHTS = {
    "old_attn_gru": 0.2,
    "new_attn_gru": 0.7,
    "new_tcn": 0.1,
}


def main() -> None:
    old_path = ROOT / "outputs" / "selector_full"
    new_path = ROOT / "outputs" / "06_selector_experiments" / "multimodel_attn_tcn_seed20260621"
    out_path = ROOT / "outputs" / "08_blend_selector"
    out_path.mkdir(parents=True, exist_ok=True)

    # --- OOF ---
    old_oof = np.load(old_path / "oof_selector_scores.npz", allow_pickle=True)
    new_oof = np.load(new_path / "oof_selector_scores.npz", allow_pickle=True)
    assert list(old_oof["candidate_names"]) == list(new_oof["candidate_names"])
    assert np.array_equal(old_oof["y"], new_oof["y"])
    covered = old_oof["covered"] & new_oof["covered"]
    blend_oof = (
        WEIGHTS["old_attn_gru"] * old_oof["attn_gru_scores"]
        + WEIGHTS["new_attn_gru"] * new_oof["attn_gru_scores"]
        + WEIGHTS["new_tcn"] * new_oof["tcn_scores"]
    )
    blend_prior = (
        WEIGHTS["old_attn_gru"] * old_oof["attn_gru_prior"]
        + WEIGHTS["new_attn_gru"] * new_oof["attn_gru_prior"]
        + WEIGHTS["new_tcn"] * new_oof["tcn_prior"]
    )
    np.savez_compressed(
        out_path / "oof_selector_scores.npz",
        covered=covered,
        y=old_oof["y"],
        cands=old_oof["cands"],
        ens_scores=blend_oof.astype(np.float32),
        ens_prior=blend_prior.astype(np.float32),
        candidate_names=old_oof["candidate_names"],
    )
    print(f"wrote oof bank: {out_path / 'oof_selector_scores.npz'}")
    # Verify against search
    cov_idx = np.flatnonzero(covered)
    m = P.search_argmax_soft_gate(old_oof["cands"][cov_idx], blend_oof[cov_idx], old_oof["y"][cov_idx])
    print(f"  blend OOF gate hit: {m['metrics']['hit']:.4f}  (target ~0.6575)")

    # --- test ---
    old_test = np.load(old_path / "test_selector_scores.npz", allow_pickle=True)
    new_test = np.load(new_path / "test_selector_scores.npz", allow_pickle=True)
    assert list(old_test["candidate_names"]) == list(new_test["candidate_names"])
    assert np.allclose(old_test["cands"], new_test["cands"])
    blend_test = (
        WEIGHTS["old_attn_gru"] * old_test["attn_gru_scores"]
        + WEIGHTS["new_attn_gru"] * new_test["attn_gru_scores"]
        + WEIGHTS["new_tcn"] * new_test["tcn_scores"]
    )
    # Old test bank only saved ens_prior (single-model collapse). Use it as the
    # old_attn_gru prior; new test bank saves both per-model priors.
    blend_test_prior = (
        WEIGHTS["old_attn_gru"] * old_test["ens_prior"]
        + WEIGHTS["new_attn_gru"] * new_test["attn_gru_prior"]
        + WEIGHTS["new_tcn"] * new_test["tcn_prior"]
    )
    np.savez_compressed(
        out_path / "test_selector_scores.npz",
        cands=old_test["cands"],
        ens_scores=blend_test.astype(np.float32),
        ens_prior=blend_test_prior.astype(np.float32),
        candidate_names=old_test["candidate_names"],
    )
    print(f"wrote test bank: {out_path / 'test_selector_scores.npz'}")

    print("\nWeights used:")
    for k, w in WEIGHTS.items():
        print(f"  {k}: {w}")


if __name__ == "__main__":
    main()
