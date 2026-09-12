from pathlib import Path

from src.champion.v353_refit_trackman_pfd_students import sha256


def test_sha256_is_uppercase_and_content_sensitive(tmp_path: Path):
    path = tmp_path / "model.txt"
    path.write_bytes(b"tree\nversion=v4\n")
    digest = sha256(path)
    assert digest == digest.upper()
    assert len(digest) == 64
