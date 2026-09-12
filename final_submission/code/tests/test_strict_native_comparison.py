from validate_strict_fresh_fit import compare_native


def test_native_model_comparison_checks_structure_and_values():
    assert compare_native({"trees": [1.,2.]}, {"trees": [1.,2.]})["within_tolerance"]
    assert not compare_native({"trees": [1.,2.]}, {"trees": [1.]})["within_tolerance"]
    assert not compare_native({"trees": [1.,2.]}, {"trees": [1.,3.]})["within_tolerance"]
    assert not compare_native({"trees": []}, {})["within_tolerance"]
