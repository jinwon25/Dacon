from __future__ import annotations

import pytest

from src.archive.v208_h1_workload_multiseed_fit import (
    model_config,
    restrictions,
)


def test_model_config_changes_only_registered_seed() -> None:
    config = model_config(43)
    assert config["seed"] == 43
    assert config["iterations"] == 1200
    assert config["depth"] == 8
    with pytest.raises(ValueError, match="unregistered seed"):
        model_config(45)


def test_v208_is_fixed_forward_confirmation_without_test_batch() -> None:
    audit = restrictions()
    assert audit["strictly_prior_season_fits"]
    assert audit["same_recipe_as_seed42"]
    assert audit["fixed_confirmation_seeds"]
    assert not audit["test_csv_read"]
    assert not audit["test_aggregate_used"]
    assert not audit["other_test_rows_required"]
    assert not audit["public_score_used_for_selection"]
