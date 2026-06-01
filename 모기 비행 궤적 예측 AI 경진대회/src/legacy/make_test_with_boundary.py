"""Generate test submission CSV using the trained 31-cand selector + boundary correction.

Usage (after selector + boundary OOF sweep done):
  python scripts/make_test_with_boundary.py CAP APPLY SEED

Writes submission_boundary_tiny_gate.csv under
outputs/03_submission_candidates/poly31_BEST_<config>/.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pipeline as P  # noqa: E402


def main() -> None:
    if len(sys.argv) != 4:
        print("usage: make_test_with_boundary.py CAP APPLY SEED")
        sys.exit(1)
    cap = float(sys.argv[1])
    apply_scale = float(sys.argv[2])
    seed = int(sys.argv[3])

    selector_out = ROOT / "outputs" / "06_selector_experiments" / "attn_gru_poly31_seed20260620"
    out = ROOT / "outputs" / "03_submission_candidates" / f"poly31_cap{cap:g}_apply{apply_scale:g}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)

    P.call_main(P.BOUNDARY_MAIN, [
        "--root", P.DATA_ROOT,
        "--out-dir", out,
        "--fold", 0, "--folds", 5,
        "--score-bank", selector_out / "oof_selector_scores.npz",
        "--make-test",
        "--test-score-bank", selector_out / "test_selector_scores.npz",
        "--epochs", 1, "--fine-epochs", 1, "--min-epochs", 1, "--patience", 1,
        "--hidden", 64, "--batch", 8192,
        "--lr", 0.001, "--fine-lr-scale", 0.18,
        "--cap", cap, "--apply-scale", apply_scale,
        "--device", "cpu", "--seed", seed, "--save-val-pred",
    ])
    print(f"\nGenerated submission_boundary_tiny_gate.csv in: {out}")


if __name__ == "__main__":
    main()
