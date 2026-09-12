from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "experiment_workbench.ipynb"


def _load_notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict[str, object]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def test_experiment_notebook_is_clean_nbformat4() -> None:
    notebook = _load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    assert notebook["cells"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_experiment_notebook_code_cells_compile() -> None:
    notebook = _load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile(_source(cell), f"{NOTEBOOK.name}:cell-{index}", "exec")


def test_experiment_notebook_dry_run_executes_end_to_end(
    monkeypatch,
) -> None:
    notebook = _load_notebook()
    namespace: dict[str, object] = {"__name__": "__notebook_test__"}
    monkeypatch.chdir(PROJECT_ROOT)

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            code = compile(_source(cell), f"{NOTEBOOK.name}:cell-{index}", "exec")
            exec(code, namespace)

    assert namespace["DRY_RUN"] is True
    assert namespace["RUN_HEAVY"] is False
    assert namespace["RUN_PACKAGING"] is False


def test_experiment_notebook_routes_to_authoritative_modules() -> None:
    notebook = _load_notebook()
    content = "\n".join(_source(cell) for cell in notebook["cells"])

    required_tokens = {
        "DRY_RUN = True",
        "RUN_HEAVY = False",
        "RUN_PACKAGING = False",
        "src.archive.v16_multiseason_screen",
        "src.archive.evaluate_v16_robust",
        "src.archive.train_v16_residual",
        "src.archive.package_v16_residual",
        "src.archive.validate_v16_residual",
        "tests/test_experiment_notebook.py",
        "1093.3213473808",
    }
    assert required_tokens <= set(token for token in required_tokens if token in content)


def test_experiment_notebook_does_not_embed_private_payloads() -> None:
    notebook = _load_notebook()
    content = "\n".join(_source(cell) for cell in notebook["cells"])
    lowered = content.lower()

    assert "control_success,pitcher_id" not in lowered
    assert "api_token" not in lowered
    assert "authorization: bearer" not in lowered
    assert len(content) < 30_000
