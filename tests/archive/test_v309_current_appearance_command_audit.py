import numpy as np

from src.archive.v309_current_appearance_command_audit import refine_denominator


def test_refine_denominator_chooses_compatible_expected_multiple() -> None:
    minimum = np.asarray([5.0, 10.0])
    success = np.asarray([0.4, 0.3])
    middle = np.asarray([0.2, 0.1])
    expected = np.asarray([21.0, 31.0])
    refined = refine_denominator(minimum, success, middle, expected, 220)
    np.testing.assert_allclose(refined, [20.0, 30.0])
