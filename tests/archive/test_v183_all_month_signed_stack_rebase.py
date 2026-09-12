import json

from src.archive.v183_all_month_signed_stack_rebase import load_all_month_weights


def test_load_all_month_weights_selects_unomitted_solution(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "protocol": "V165_SIGNED_GROUP_CONSTRAINED_STACK_V1",
                "source_trials": [
                    {
                        "variant": "all_months",
                        "omitted_month_constraint": None,
                        "source_gate_passed": True,
                        "net_weights": '{"a": 0.2, "b": -0.1}',
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert load_all_month_weights(path) == {"a": 0.2, "b": -0.1}
