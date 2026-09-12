import pandas as pd
from src import corrected_top1100_features as corrected
from src import top1100_features as legacy
from src import advanced_domain_features, corrected_state_residual_oof


def test_corrected_and_legacy_endpoint_recipes_stay_separate():
    history = pd.DataFrame({"pitcher_id": [1,1], "season": [2019,2020],
                            "asof_pitcher_n": [10,30],
                            "asof_pitcher_success_rate": [0.5,0.6]})
    query = pd.DataFrame({"pitcher_id": [1], "season": [2021]})
    def baseline(module):
        prior = module._prior_table(history, "pitcher_id", "asof_pitcher_n",
                                    "asof_pitcher_success_rate")
        return module._merge_prior(query, prior, "pitcher_id")["__baseline_n"].iloc[0]
    assert baseline(corrected) == 30
    assert baseline(legacy) == 10


def test_corrected_trainers_use_corrected_features():
    assert advanced_domain_features.build_features is corrected.build_features
    assert corrected_state_residual_oof.build_features is corrected.build_features
