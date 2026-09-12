import pandas as pd

from src.multi_year_state_model import _add_categories


def test_category_builder_does_not_add_target():
    train = pd.DataFrame(
        {
            "game_dayofweek": [1], "top_bottom": ["T"], "game_type": ["R"],
            "domain3": ["R_CORE"], "base_state": ["___"], "pitcher_hand": [1],
            "batter_hand": [2], "pitcher_team_id": [3], "batter_team_id": [4],
        }
    )
    result = _add_categories(train, pd.DataFrame({"safe": [0.5]}))
    assert "control_success" not in result
    assert all(str(result[column].dtype) == "category" for column in result if column.startswith("cat__"))
