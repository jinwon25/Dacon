from pathlib import Path


def test_parent_base_recipe_is_frozen():
    import train_parent_base as module

    assert module.NUM_ITERATIONS == 54
    assert module.VARIANT["name"] == "lgb_engineered_l31"
    assert module.VARIANT["num_leaves"] == 31
    assert module.VARIANT["min_data_in_leaf"] == 500


def test_parent_validator_covers_base_assets():
    import validate_parent_native_fit as module

    assert module.COMPONENTS["base"] == (
        "lgb_model.txt",
        "feature_spec.json",
        "rf_model.joblib",
    )
    assert Path(module.__file__).is_file()


def test_parent_trackman_recipe_is_frozen():
    import train_parent_trackman as module

    assert module.NUM_ITERATIONS == 41
    assert set(module.EXPECTED) == {"train.csv", "trackman_history.csv"}


def test_parent_recency_rf_uses_fixed_training_source():
    import train_parent_recency_rf as module

    assert module.EXPECTED_TRAIN.startswith("d2081186")


def test_parent_validator_normalizes_ridge_and_pandas_state():
    import numpy as np
    import pandas as pd
    import validate_parent_native_fit as module

    value = {"x": np.asarray([1.0, np.nan]), "s": pd.Series([2.0])}
    plain = module._plain(value)
    assert plain["x"] == [1.0, None]
    assert plain["s"]["values"] == [2.0]


def test_parent_validator_covers_newly_verified_parent_assets():
    import validate_parent_native_fit as module

    assert len(module.COMPONENTS["v14_refinement"]) == 4
    assert len(module.COMPONENTS["joint_state_mode"]) == 14
    assert module.COMPONENTS["v20_pfd"][:2] == (
        "v20_pfd_control.txt",
        "v20_pfd_soft_l050.txt",
    )
    assert module.COMPONENTS["v25_anchor"] == (
        "v25_postbreak_anchor.joblib",
        "v25_postbreak_anchor_spec.json",
    )


def test_parent_validator_normalizes_bspline_state():
    import numpy as np
    from scipy.interpolate import BSpline
    import validate_parent_native_fit as module

    spline = BSpline(
        np.asarray([0.0, 0.0, 1.0, 1.0]),
        np.asarray([1.0, 2.0]),
        1,
    )
    plain = module._plain(spline)
    assert plain["class"] == "BSpline"
    assert plain["k"] == 1
    assert plain["c"] == [1.0, 2.0]
