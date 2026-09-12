from src.archive.package_v28_anchor_hand_route import NEW_BLOCK, OLD_BLOCK, _patch_script


def test_patch_replaces_only_the_frozen_eta_block() -> None:
    source = "before\n" + OLD_BLOCK + "after\n"
    patched = _patch_script(source)
    assert OLD_BLOCK not in patched
    assert NEW_BLOCK in patched
    assert patched.startswith("before\n")
    assert patched.endswith("after\n")
