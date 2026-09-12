import unittest

import numpy as np

from src.metrics import evaluate_competition, evaluate_group


class MetricTests(unittest.TestCase):
    def test_ficr_is_weighted_by_actual_generation(self) -> None:
        actual = np.array([100.0, 900.0])
        forecast = np.array([100.0, 830.0])

        result = evaluate_group(actual, forecast, capacity=1_000.0)

        error_rate = np.abs(actual - forecast) / 1_000.0
        unit_price = np.where(error_rate <= 0.06, 4.0, np.where(error_rate <= 0.08, 3.0, 0.0))
        time_mean_ficr = unit_price.mean() / 4.0

        self.assertAlmostEqual(result.nmae, 0.035)
        self.assertAlmostEqual(result.one_minus_nmae, 0.965)
        self.assertAlmostEqual(result.ficr, 0.775)
        self.assertAlmostEqual(time_mean_ficr, 0.875)
        self.assertNotAlmostEqual(result.ficr, time_mean_ficr)
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

    def test_nan_actual_is_excluded_but_capacity_boundary_is_included(self) -> None:
        actual = np.array([np.nan, 99.999, 100.0])
        forecast = np.array([0.0, 99.999, 100.0])
        result = evaluate_group(actual, forecast, capacity=1_000.0)
        self.assertEqual(result.n_samples, 1)
        self.assertAlmostEqual(result.score, 1.0)

    def test_no_eligible_rows_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "No valid evaluation rows"):
            evaluate_group(
                np.array([np.nan, 0.0, 99.999]),
                np.array([0.0, 0.0, 99.999]),
                capacity=1_000.0,
            )


if __name__ == "__main__":
    unittest.main()
