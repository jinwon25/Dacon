import json
from pathlib import Path

import numpy as np
from pandas.testing import assert_frame_equal

import script
from src.data import read_main
from src.features import FeatureBuilder


PROJECT = Path(__file__).resolve().parents[1]


def test_training_and_submission_feature_builders_match():
    sample = read_main(PROJECT / "data" / "test.csv")
    spec = json.loads((PROJECT / "model" / "feature_spec.json").read_text(encoding="utf-8"))
    training_builder = FeatureBuilder.from_spec(spec)
    expected = training_builder.transform(sample)
    actual = script.build_features(sample, spec)
    assert_frame_equal(actual, expected, check_dtype=True)


def test_sample_predictions_are_finite_probabilities():
    sample = read_main(PROJECT / "data" / "test.csv")
    prediction = script.predict_dataframe(sample)
    assert len(prediction) == len(sample)
    assert np.isfinite(prediction).all()
    assert ((prediction >= 0.0) & (prediction <= 1.0)).all()
