import numpy as np
import pandas as pd

from src.audit_v352_trackman_pfd import predict_partitioned


class RowLocalModule:
    @staticmethod
    def predict_dataframe(frame: pd.DataFrame) -> np.ndarray:
        return frame["x"].to_numpy(np.float64) * 0.5


def test_predict_partitioned_preserves_row_local_predictions():
    frame = pd.DataFrame({"x": np.arange(7, dtype=float)})
    np.testing.assert_allclose(
        predict_partitioned(RowLocalModule(), frame),
        RowLocalModule.predict_dataframe(frame),
    )
