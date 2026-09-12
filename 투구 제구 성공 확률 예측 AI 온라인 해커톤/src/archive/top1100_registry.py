"""Write the Top-1100 research debt audit and preregistered experiment registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha(path: Path) -> str:
    digest = hashlib.sha256(); digest.update(path.read_bytes()); return digest.hexdigest().upper()


def run(project: Path) -> None:
    report_dir = project / "research/reports/top1100"; report_dir.mkdir(parents=True, exist_ok=True)
    parent = project / "submit_v2.zip"
    debt = pd.DataFrame([
        {"debt": "exact V2 row-level component OOF", "status": "FAIL", "evidence": "strict fixed-fit nested OOF was not completed; frozen cache is diagnostic only"},
        {"debt": "outer early stopping removed and executed", "status": "FAIL", "evidence": "implementation exists in src/nested_v2.py/nested_fast.py but full run exceeded local runtime"},
        {"debt": "2024 virgin holdout wording", "status": "PASS", "evidence": "all reports call 2024 locked confirmation and document repeated reuse"},
        {"debt": "V4/V5 clean ablation interpretation", "status": "PASS", "evidence": "submission_audit and frozen decomposition label branch precedence"},
        {"debt": "RF categorical ID treatment", "status": "FAIL", "evidence": "new FM/CatBoost family required before using RF as structural reference"},
        {"debt": "fair CatBoost comparison", "status": "FAIL", "evidence": "legacy CatBoost used outer callback/complex hierarchy; new fair pilot not yet run"},
        {"debt": "legacy target encoder isolation", "status": "PASS", "evidence": "permutation/current-label/future-label audit; excluded from Top-1100 recipes"},
        {"debt": "Trackman confidence is identity accuracy", "status": "PASS", "evidence": "reports distinguish stability from identity and keep low confidence on fallback"},
        {"debt": "hand mapping target-free two-way comparison", "status": "FAIL", "evidence": "structural alignment audit cannot identify mapping; no physical profile promotion"},
        {"debt": "plate coordinate availability", "status": "PASS", "evidence": "schema has no plate_x/plate_z; no plate-location model is claimed"},
    ])
    debt.to_csv(report_dir / "research_debt_audit.csv", index=False)
    (report_dir / "research_debt_audit.md").write_text("# Research debt audit\n\n" + "\n".join(f"- **{row.status}** `{row.debt}` — {row.evidence}" for row in debt.itertuples()) + "\n", encoding="utf-8")
    (report_dir / "assumption_changes.md").write_text("""# Assumption changes

- The formal local `test.csv` contains 5 sample rows; the approximately 245,789-row evaluation batch is server-side and is not reconstructed from test order or distribution.
- The direct adjacent-row as-of check was replaced by a pitcher-group trajectory check. It confirms `asof_pitcher_n` increments by one for 1,473,508 transitions; displayed rate precision gives a 99.935% transition match within 0.01 count units, so integer counts are treated as rounded intervals rather than exact labels.
- Target-free game-block alignment found 7,228 main blocks and 5,980 Trackman games, with 4,810 calendar-bucket assignments but weak distance/placebo separation. It is a candidate generator, not an approved player identity map.
- Full honest nested OOF remains blocked by local runtime. No score or leaderboard claim is derived from a partial run.
""", encoding="utf-8")
    configs = [
        ("P0A_asof_state", "Cumulative asof statistics recover season exposure and improve recent-vs-career separation", "state", {"source":"asof_*","target_free":True}),
        ("P0B_structural_alignment", "Legal game/count structure can narrow main-Trackman identity candidates", "alignment", {"distance":"calendar+state+length","placebo_repeats":20}),
        ("F1_fm_pilot_01", "Low-rank pitcher/batter/team x context interactions improve resolution", "FM", {"emb_dim":8,"dropout":0.1,"features":"P0A+context"}),
        ("F1_fm_pilot_02", "Higher embedding capacity shares rare interactions", "FM", {"emb_dim":16,"dropout":0.1,"features":"P0A+context"}),
        ("F1_fm_pilot_03", "Small DCNv2 branch adds higher-order cross signal", "DeepFM/DCNv2", {"emb_dim":16,"cross_layers":2,"features":"P0A+context"}),
        ("F2_dynamic_logit_01", "Joint partial pooling beats independent frequency backoff", "dynamic_logit", {"random_effects":"pitcher,team,count_platoon","offset":"none"}),
        ("F2_dynamic_logit_02", "V2 offset stabilizes a dynamic multilevel residual", "dynamic_logit", {"random_effects":"pitcher,team,count_platoon","offset":"v2_logit"}),
        ("F2_dynamic_logit_03", "Discounted recent state captures role/form drift", "dynamic_logit", {"discount_grid":[0.8,0.9,0.97],"offset":"v2_logit"}),
        ("F3_catboost_01", "Categorical-safe CatBoost is a fair challenger when outer iteration is fixed", "CatBoost", {"depth":5,"iterations":300,"learning_rate":0.03,"l2_leaf_reg":30,"categorical":"IDs,hand,count,base,game_type"}),
        ("F4_realmlp_01", "PLE + small entity embedding gives orthogonal nonlinear resolution", "RealMLP", {"blocks":3,"width":128,"seed":42,"metric":"brier"}),
        ("F4_tabm_01", "Parameter-efficient MLP members improve probability average", "TabM", {"members":16,"width":128,"blocks":2,"seed":42}),
        ("F5_retrieval_01", "Train-only constrained prototypes add local residual information", "retrieval_residual", {"prototypes_per_bucket":32,"max_neighbors":32,"reference":"train_only"}),
        ("F6_horizon_01", "Long/mid/short/cold experts adapt to history confidence", "horizon_MoE", {"experts":["long","mid","short","cold"],"gate":"previous_origin_OOF"}),
        ("F7_teacher_01", "Time-forward teacher soft labels can distill an orthogonal student", "teacher_distill", {"teacher":"TabICLv2","student":"small_GBDT","max_context_rows":100000}),
    ]
    columns = ["experiment_id","hypothesis","family","exact_config","train_period","inner_period","outer_period","artifact_hashes","brier","score_equivalent","logloss","reliability","resolution","weighted_delta","worst_fold_delta","seed_variance","cluster_bootstrap_ci","subgroup_worst_case","inference_seconds","peak_rss_mb","model_size_mb","zip_size_mb","gate_result","termination_reason"]
    rows = []
    for experiment_id, hypothesis, family, config in configs:
        rows.append({"experiment_id":experiment_id,"hypothesis":hypothesis,"family":family,"exact_config":json.dumps(config, sort_keys=True),"train_period":"2019-2023","inner_period":"previous-season rolling folds","outer_period":"2021-2024","artifact_hashes":json.dumps({"parent_v2":_sha(parent)}, sort_keys=True),"brier":"","score_equivalent":"","logloss":"","reliability":"","resolution":"","weighted_delta":"","worst_fold_delta":"","seed_variance":"","cluster_bootstrap_ci":"","subgroup_worst_case":"","inference_seconds":"","peak_rss_mb":"","model_size_mb":"","zip_size_mb":"","gate_result":"PREREGISTERED","termination_reason":"pending strict OOF"})
    pd.DataFrame(rows, columns=columns).to_csv(report_dir / "experiment_registry.csv", index=False)
    (report_dir / "score_ladder.md").write_text("""# Score ladder

At base rate near 0.5, approximate score-equivalent improvements are:

| ΔBrier | Score gain |
|---:|---:|
| -0.00005 | +20 |
| -0.00010 | +40 |
| -0.00025 | +100 |
| -0.00050 | +200 |
| -0.00084 | +336 |

The 763.27→1100 gap therefore requires roughly `-0.00084` Brier, not a single calibration or blend-weight adjustment. The first promotion gate is fixed at 2024 Δ≤-0.00010, recency-weighted four-fold Δ≤-0.00010, worst fold≤+0.00005 and pitcher-season bootstrap negative probability≥0.95.
""", encoding="utf-8")
    print(f"registry_rows={len(rows)} parent_sha={_sha(parent)}")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--project-dir",type=Path,default=Path(".")); args=parser.parse_args(); run(args.project_dir.resolve())


if __name__=="__main__": main()
