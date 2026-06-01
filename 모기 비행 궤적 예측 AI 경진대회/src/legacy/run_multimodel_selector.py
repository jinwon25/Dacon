"""Phase A: 5-fold multi-model selector ensemble on the original 27-candidate bank.

Trains attn_gru + tcn in the same run; their per-fold OOF scores are
mean-ensembled by the existing training loop. Seed 20260621 to keep this run
distinct from prior selector_full (attn_gru-only) and the poly31 attempt.

Decision rule:
- 5-fold OOF ensemble gate >= 0.6650 -> proceed to boundary OOF sweep.
- 0.6620-0.6649 -> consider adding BiGRU as a third member.
- < 0.6620 -> abandon Phase A, try Transformer encoder instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pipeline as P  # noqa: E402


def main() -> None:
    out_dir = ROOT / "outputs" / "06_selector_experiments" / "multimodel_attn_tcn_seed20260621"
    P.call_main(P.SELECTOR_MAIN, [
        "--root", P.DATA_ROOT,
        "--out-dir", out_dir,
        "--models", "attn_gru", "tcn",
        "--folds", 5,
        "--pre-epochs", 14, "--fine-epochs", 10, "--freeze-fine-epochs", 3,
        "--epoch-plus", 10, "--min-epochs", 5, "--patience", 8,
        "--hidden", 96, "--batch", 1024,
        "--lr", 0.001, "--fine-lr-scale", 0.12,
        "--prior-strength", 0.65, "--regime-prior-strength", 0.45,
        "--pairwise-loss-weight", 0.25, "--pairwise-margin", 0.12, "--pairwise-min-label-gap", 0.04,
        "--fine-distill-weight", 0.55, "--fine-distill-temp", 0.07,
        "--reverse-pretrain", "--norm-real-only",
        "--device", "cpu", "--seed", 20260621, "--log-every", 1,
    ])
    P.write_selector_score_variants(out_dir)
    P.print_selector_summary(out_dir)


if __name__ == "__main__":
    main()
