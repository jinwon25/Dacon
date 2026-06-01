"""Phase D: 5-fold Transformer selector on the original 27-candidate bank.

Architecture: 2-layer Transformer encoder (4 heads, dim_feedforward=2*hidden,
norm_first) over the 6-step physics-summary sequence + cross-attention from
each candidate query. Mirrors CandidateAttentionGRUSelector outputs so the
existing selector training loop, priors, distillation, and ensemble code path
work unchanged.

Seed 20260622 to keep this run distinct from prior selector_full and the
poly31/multimodel attempts. Decision rule:
- 5-fold OOF gate >= 0.6620 (selector_full was 0.6561) -> proceed to ensemble.
- 0.6580-0.6619 -> ensemble with old attn_gru anyway.
- < 0.6580 -> mark Transformer as failed for this dataset size.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pipeline as P  # noqa: E402


def main() -> None:
    out_dir = ROOT / "outputs" / "06_selector_experiments" / "transformer_seed20260622"
    P.call_main(P.SELECTOR_MAIN, [
        "--root", P.DATA_ROOT,
        "--out-dir", out_dir,
        "--models", "transformer",
        "--folds", 5,
        "--pre-epochs", 14, "--fine-epochs", 10, "--freeze-fine-epochs", 3,
        "--epoch-plus", 6, "--min-epochs", 5, "--patience", 8,
        "--hidden", 64, "--batch", 2048,
        "--lr", 0.001, "--fine-lr-scale", 0.12,
        "--prior-strength", 0.65, "--regime-prior-strength", 0.45,
        "--pairwise-loss-weight", 0.25, "--pairwise-margin", 0.12, "--pairwise-min-label-gap", 0.04,
        "--fine-distill-weight", 0.55, "--fine-distill-temp", 0.07,
        "--reverse-pretrain", "--norm-real-only",
        "--device", "cpu", "--seed", 20260622, "--log-every", 1,
    ])
    P.write_selector_score_variants(out_dir)
    P.print_selector_summary(out_dir)


if __name__ == "__main__":
    main()
