import numpy as np
import pandas as pd

from src.champion.v354_build_late_hierarchy_package import (
    BUNDLE_MEMBER,
    FUNCTION_ANCHOR,
    NEW_TAIL,
    OLD_TAIL,
    bundle_zip_info,
    patch_script,
)


def test_patch_script_adds_one_late_hierarchy_runtime():
    source = FUNCTION_ANCHOR + "]\n\n" + OLD_TAIL
    patched = patch_script(source)
    assert patched.count("def _predict_late_hierarchy") == 1
    assert NEW_TAIL in patched
    assert OLD_TAIL not in patched


def test_bundle_member_has_fixed_timestamp():
    info = bundle_zip_info()
    assert info.filename == BUNDLE_MEMBER
    assert info.date_time == (2026, 9, 1, 0, 0, 0)
