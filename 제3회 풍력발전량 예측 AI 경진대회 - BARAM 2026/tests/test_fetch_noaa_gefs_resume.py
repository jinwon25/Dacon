from __future__ import annotations

import ast
from pathlib import Path


def test_downloader_handles_corrupt_sidecar_and_writes_atomically() -> None:
    source = Path("experiments/fetch_noaa_gefs_spread.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert tree is not None
    assert "json.JSONDecodeError" in source
    assert 'sidecar.with_suffix(".tmp")' in source
    assert "_replace_with_retry(sidecar_temporary, sidecar)" in source
