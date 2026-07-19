from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_service.adapters import adapt_report, meta_gate_sweep_evaluation
from agent_service.cli import build_parser, main
from agent_service.policy import PromotionPolicy


POLICY = {
    "version": "test-v1",
    "min_locked_score_delta": 0.00015,
    "min_locked_one_minus_nmae_delta": 0.0,
    "min_locked_ficr_delta": 0.0,
    "min_expected_macro_score_delta": 0.00005,
    "min_positive_month_fraction": 0.5,
    "min_bootstrap_positive_fraction": 0.8,
    "min_bootstrap_q05": -0.00025,
    "max_changed_ratio": 0.25,
    "max_p95_movement_ratio": 0.015,
    "failed_family_max_coverage_fraction": 0.25,
}


def sweep_report() -> dict[str, object]:
    monthly = {
        str(month): {
            "score": 0.0002 if month < 11 else -0.0001,
            "one_minus_nmae": 0.00001,
            "ficr": 0.00039 if month < 11 else -0.00021,
        }
        for month in range(7, 13)
    }
    return {
        "locked_incremental_over_reference": {
            "reference_policy": {"threshold": 0.55, "alpha": 0.25},
            "selected_policy": {"threshold": 0.545, "alpha": 0.5},
            "metrics": {
                "delta": {
                    "score": 0.00063886,
                    "one_minus_nmae": 0.00002160,
                    "ficr": 0.00125612,
                }
            },
            "monthly_deltas": monthly,
            "positive_months": 4,
            "day_bootstrap": {
                "positive_fraction": 0.8735,
                "q05": -0.000312759,
            },
        },
        "submission": {
            "changed_ratio": 0.11929,
            "p95_absolute_movement_kwh": 234.745,
        },
    }


def test_sweep_adapter_uses_incremental_incumbent_comparison() -> None:
    evaluation = meta_gate_sweep_evaluation(sweep_report())
    assert evaluation.family == "exact_oof_meta_gate_sweep"
    assert evaluation.locked_score_delta == pytest.approx(0.00063886)
    assert evaluation.expected_macro_score_delta == pytest.approx(0.00063886 / 3.0)
    assert evaluation.positive_months == 4
    assert evaluation.total_months == 6
    assert evaluation.leakage_risk == "low"
    decision = PromotionPolicy(POLICY).evaluate(evaluation)
    assert decision.outcome == "rejected"
    assert "day-bootstrap lower tail is too negative" in decision.reasons


def test_sweep_can_pass_policy_when_paired_lower_tail_is_stable() -> None:
    report = sweep_report()
    report["locked_incremental_over_reference"]["day_bootstrap"]["q05"] = -0.00020
    evaluation = meta_gate_sweep_evaluation(report)
    assert PromotionPolicy(POLICY).evaluate(evaluation).outcome == "candidate"


def test_sweep_adapter_requires_a_generated_candidate() -> None:
    report = sweep_report()
    report["submission"] = None
    with pytest.raises(ValueError, match="no generated submission"):
        adapt_report("meta_gate_sweep", report)


def test_report_adapt_cli_is_database_free(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report_path = tmp_path / "sweep.json"
    output_path = tmp_path / "evaluation.json"
    report_path.write_text(json.dumps(sweep_report()), encoding="utf-8")

    # No .agents config exists under tmp_path. The pure adapter command must still
    # work and must not create or initialize a service database.
    main(
        [
            "--root",
            str(tmp_path),
            "report-adapt",
            "meta_gate_sweep",
            str(report_path),
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["family"] == "exact_oof_meta_gate_sweep"
    assert not (tmp_path / "artifacts_final").exists()
    assert "meta_gate_sweep" in capsys.readouterr().out


def test_run_evaluate_cli_accepts_report_adapter_pair() -> None:
    args = build_parser().parse_args(
        [
            "run-evaluate",
            "7",
            "--report",
            "sweep.json",
            "--adapter",
            "meta_gate_sweep",
        ]
    )
    assert args.run_id == 7
    assert args.report == "sweep.json"
    assert args.adapter == "meta_gate_sweep"
