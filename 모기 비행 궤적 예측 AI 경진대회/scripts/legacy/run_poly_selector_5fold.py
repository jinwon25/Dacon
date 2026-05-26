"""Production 5-fold selector training for the 31-candidate (poly-extended) bank.

Mirrors the production hyperparameters used in NEXT_RUN_2026-05-10_perpm040.md,
with seed 20260620 to distinguish from the failed perp=-0.40 attempts.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pipeline as P  # noqa: E402


def main() -> None:
    out_dir = ROOT / "outputs" / "06_selector_experiments" / "attn_gru_poly31_seed20260620"
    P.call_main(P.SELECTOR_MAIN, [
        "--root", P.DATA_ROOT,
        "--out-dir", out_dir,
        "--models", "attn_gru",
        "--folds", 5,
        "--pre-epochs", 14, "--fine-epochs", 10, "--freeze-fine-epochs", 3,
        "--epoch-plus", 10, "--min-epochs", 5, "--patience", 8,
        "--hidden", 96, "--batch", 1024,
        "--lr", 0.001, "--fine-lr-scale", 0.12,
        "--prior-strength", 0.65, "--regime-prior-strength", 0.45,
        "--pairwise-loss-weight", 0.25, "--pairwise-margin", 0.12, "--pairwise-min-label-gap", 0.04,
        "--fine-distill-weight", 0.55, "--fine-distill-temp", 0.07,
        "--reverse-pretrain", "--norm-real-only",
        "--device", "cpu", "--seed", 20260620, "--log-every", 1,
    ])
    P.write_selector_score_variants(out_dir)
    P.print_selector_summary(out_dir)


if __name__ == "__main__":
    main()
