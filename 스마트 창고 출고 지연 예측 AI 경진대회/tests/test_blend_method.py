"""A pinned final recipe must not be replaced by a lower-OOF alternative."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_pinned_slsqp_and_automatic_selection_are_distinct(tmp_path):
    predictions = np.array([[1., 9.], [9., 1.], [4., 16.]])
    target = np.expm1(np.log1p(predictions).mean(axis=1))
    dirs = []
    for i in range(2):
        folder = tmp_path / f'model{i}'
        folder.mkdir()
        pd.DataFrame({'ID': [1, 2, 3], 'avg_delay_minutes_next_30m': target, 'pred': predictions[:, i]}).to_csv(folder / 'oof_predictions.csv', index=False)
        pd.DataFrame({'ID': [11, 12, 13], 'avg_delay_minutes_next_30m': predictions[:, i]}).to_csv(folder / 'submission.csv', index=False)
        dirs.append(str(folder))
    script = Path(__file__).resolve().parents[1] / 'src/blend_safe.py'
    for method in ('auto', 'global_slsqp'):
        out = tmp_path / method
        subprocess.run([sys.executable, str(script), '--model-dirs', *dirs, '--method', method, '--output-dir', str(out)], check=True, capture_output=True)
        metadata = json.loads((out / 'blend_metadata.json').read_text())
        assert metadata['selection_mode'] == method
        assert metadata['best_method'] == ('geometric' if method == 'auto' else method)
        result = pd.read_csv(out / 'submission.csv')
        assert result['ID'].tolist() == [11, 12, 13]
        if method == 'auto':
            np.testing.assert_allclose(result['avg_delay_minutes_next_30m'], target)
        else:
            weights = np.array(metadata['all_results']['global_slsqp']['weights'])
            np.testing.assert_allclose(result['avg_delay_minutes_next_30m'], predictions @ weights)
