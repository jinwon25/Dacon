"""Publication regressions that the previous suffix-only scanner missed."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('publication', Path(__file__).resolve().parents[1] / 'scripts/audit_public_repository.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class PublicationAuditTests(unittest.TestCase):
    def test_fitted_json_is_rejected_even_under_an_arbitrary_name(self):
        self.assertIn('trained XGBoost JSON', module.findings('src/assets/model.json', b'{"learner":{"gradient_booster":{}}}'))

    def test_identity_csv_is_rejected_inside_report_directory(self):
        self.assertIn('row identifiers or targets in CSV', module.findings('project/reports/mapping.csv', b'pitcher_id,pitcher_trackman_id\n1,2\n'))

    def test_aggregate_metrics_are_allowed(self):
        self.assertEqual([], module.findings('project/research/reports/metrics.csv', b'version,public_score\nv1,100\n'))

    def test_prediction_csv_is_rejected(self):
        self.assertTrue(module.findings('project/submissions/final.csv', b'ID,pred\n1,0.5\n'))

    def test_notebook_outputs_are_rejected(self):
        self.assertIn('notebook execution output', module.findings('analysis.ipynb', b'{"cells":[{"cell_type":"code","execution_count":1,"outputs":[{"text":"data"}]}]}'))

    def test_example_environment_is_allowed(self):
        self.assertEqual([], module.findings('.env.example', b'DACON_TOKEN=\n'))

if __name__ == '__main__':
    unittest.main()
