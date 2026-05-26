"""1-fold low-epoch smoke test for the updated 31-candidate selector.

Confirms make_candidates, build_samples, full training, and OOF eval all
run without error on the new candidate bank.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pipeline as P  # noqa: E402


def main() -> None:
    out_dir = ROOT / "outputs" / "06_selector_experiments" / "smoke_poly31_seed20260620"
    P.call_main(P.SELECTOR_MAIN, [
        "--root", P.DATA_ROOT,
        "--out-dir", out_dir,
        "--models", "attn_gru",
        "--folds", 5, "--fold-limit", 1, "--skip-full",
        "--pre-epochs", 2, "--fine-epochs", 2, "--freeze-fine-epochs", 1,
        "--epoch-plus", 0, "--min-epochs", 1, "--patience", 2,
        "--hidden", 48, "--batch", 2048,
        "--lr", 0.001, "--fine-lr-scale", 0.12,
        "--prior-strength", 0.65, "--regime-prior-strength", 0.45,
        "--pairwise-loss-weight", 0.25, "--pairwise-margin", 0.12, "--pairwise-min-label-gap", 0.04,
        "--fine-distill-weight", 0.55, "--fine-distill-temp", 0.07,
        "--reverse-pretrain", "--norm-real-only",
        "--device", "cpu", "--seed", 20260620, "--log-every", 1,
    ])


if __name__ == "__main__":
    main()
