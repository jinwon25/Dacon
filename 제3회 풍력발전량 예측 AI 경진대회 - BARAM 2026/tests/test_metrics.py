import unittest

import numpy as np

from src.metrics import evaluate_competition, evaluate_group


class MetricTests(unittest.TestCase):
    def test_ficr_is_weighted_by_actual_generation(self) -> None:
        actual = np.array([100.0, 900.0])
        forecast = np.array([100.0, 830.0])

        result = evaluate_group(actual, forecast, capacity=1_000.0)

        self.assertAlmostEqual(result.nmae, 0.035)
        self.assertAlmostEqual(result.one_minus_nmae, 0.965)
        self.assertAlmostEqual(result.ficr, 0.775)
        self.assertAlmostEqual(result.score, 0.87)

    def test_rows_below_capacity_threshold_are_excluded(self) -> None:
        actual = np.array([99.0, 100.0, 900.0])
        forecast = np.array([99.0, 100.0, 830.0])

        result = evaluate_group(actual, forecast, capacity=1_000.0)

        self.assertEqual(result.n_samples, 2)
        self.assertAlmostEqual(result.ficr, 0.775)

    def test_competition_metric_averages_group_metrics(self) -> None:
        actual = {
            "kpx_group_1": np.array([2_160.0]),
            "kpx_group_2": np.array([2_160.0]),
            "kpx_group_3": np.array([2_100.0]),
        }
        forecast = {
            "kpx_group_1": np.array([2_160.0]),
            "kpx_group_2": np.array([2_160.0]),
            "kpx_group_3": np.array([2_100.0]),
        }

        result = evaluate_competition(actual, forecast)

        self.assertAlmostEqual(result["one_minus_nmae"], 1.0)
        self.assertAlmostEqual(result["ficr"], 1.0)
        self.assertAlmostEqual(result["score"], 1.0)

    def test_nonfinite_eligible_prediction_is_not_silently_excluded(self) -> None:
        actual = np.array([100.0, 900.0])
        forecast = np.array([100.0, np.nan])

        with self.assertRaisesRegex(ValueError, "non-finite"):
            evaluate_group(actual, forecast, capacity=1_000.0)

    def test_ficr_thresholds_are_inclusive(self) -> None:
        actual = np.array([500.0, 500.0, 500.0])
        forecast = np.array([440.0, 420.0, 419.999])

        result = evaluate_group(actual, forecast, capacity=1_000.0)

        # Exact 6% earns 4, exact 8% earns 3, and just beyond 8% earns 0.
        self.assertAlmostEqual(result.ficr, 7.0 / 12.0)


if __name__ == "__main__":
    unittest.main()
