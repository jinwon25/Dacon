from src.archive.v192_soft_loo_triyear_stack import soft_budget_table


def test_soft_budget_gate_requires_every_held_axis() -> None:
    good = {
        "gain": 0.1,
        "positive_month_fraction": 0.5,
        "worst_month_gain": -0.2,
        "minimum_domain_gain": 0.0,
    }
    bad = {**good, "gain": -0.01}
    summary = {
        "budget_ranking": [{"budget": 0.1}, {"budget": 0.2}],
        "leave_one_year_detail": {
            "b0.1__held_full_2022": {"metrics": good},
            "b0.1__held_late_2023": {"metrics": good},
            "b0.1__held_full_2024": {"metrics": good},
            "b0.2__held_full_2022": {"metrics": good},
            "b0.2__held_late_2023": {"metrics": bad},
            "b0.2__held_full_2024": {"metrics": good},
        },
    }
    table = soft_budget_table(summary)
    assert bool(table.loc[table["budget"].eq(0.1), "soft_loo_passed"].iloc[0])
    assert not bool(table.loc[table["budget"].eq(0.2), "soft_loo_passed"].iloc[0])
