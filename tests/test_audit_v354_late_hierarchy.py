import numpy as np
import pandas as pd

from src.audit_v354_late_hierarchy import predict_partitioned


class RowLocalModule:
    @staticmethod
    def predict_dataframe(frame: pd.DataFrame) -> np.ndarray:
        return frame["x"].to_numpy(np.float64) / 10.0


def test_predict_partitioned_is_row_local():
    frame = pd.DataFrame({"x": np.arange(9, dtype=float)})
    np.testing.assert_allclose(
        predict_partitioned(RowLocalModule(), frame),
        RowLocalModule.predict_dataframe(frame),
    )
