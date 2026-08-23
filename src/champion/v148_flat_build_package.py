"""Build the flat single-generation v148 package.

The original ``submit_v148.zip`` is self-contained but nests five generations of
packages inside each other and wires them together with
``importlib.util.spec_from_file_location`` plus post-load ``MODEL_DIR``
mutation.  This module rewrites that layout into

    script.py / requirements.txt / lib/*.py / model/<component>/

without touching a single model weight and without changing any prediction.
Every source transformation below is anchored on an exact string and asserted,
so an upstream edit fails the build loudly instead of silently drifting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


PROTOCOL = "V148_FLAT_SINGLE_GENERATION_PACKAGE_V1"

ORIGINAL_SHA256 = "7A27BE5878A79934544C741F283C139D40FB20484D52DB494928BCBE27E1E337"

# Original member prefix -> flat destination prefix.  Model artifacts are copied
# byte-for-byte; only their path changes.  Each destination is named after the
# ``lib`` module that loads it, so the owner of every weight is obvious.
#
# Individual weight *filenames* under ``model/base_ensemble/`` keep their
# historical prefixes on purpose: ``hybrid.json``,
# ``v20_target1160_spec.json`` and ``v25_postbreak_anchor_spec.json`` are
# themselves model artifacts that reference nine of those filenames by value,
# so renaming the files would require editing artifacts and would forfeit the
# 78/78 byte-identity guarantee.  See the refactor report.
MODEL_MAP: tuple[tuple[str, str], ...] = (
    ("model/v124/model/parent/parent/champion/", "model/base_ensemble/"),
    ("model/v124/model/parent/parent/strict/", "model/strict_asof/"),
    ("model/v124/model/conditional/", "model/conditional/"),
    ("model/v124/model/r_fm/", "model/regular_fm/"),
    ("model/v124/model/parent/v56_fm/", "model/futures_fm/"),
    ("model/h1/model/rf.pkl", "model/form_context_rf/rf.pkl"),
    ("model/c3_sign_all.joblib", "model/c3_sign_all.joblib"),
)

# Members deliberately dropped: superseded entry points and the two vestigial
# nested requirements files (see report for the version-conflict note).
DROPPED = (
    "script.py",
    "requirements.txt",
    "model/h1/script.py",
    "model/h1/requirements.txt",
    "model/v124/script.py",
    "model/v124/requirements.txt",
    "model/v124/model/components/parent_script.py",
    "model/v124/model/components/v104_features.py",
    "model/v124/model/parent/components/parent_script.py",
    "model/v124/model/parent/parent/components/champion_script.py",
    "model/v124/model/parent/parent/components/strict_script.py",
)

PATHS_MODULE = '''"""Single source of truth for artifact locations in the flat layout."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = PACKAGE_ROOT / "model"
'''

INIT_MODULE = '"""Frozen inference components (flat single-generation layout)."""\n'

ARCHITECTURE_DOC = """# Package architecture

One entry point, one generation, no nested packages.  `script.py` is the only
file the evaluation server runs; every component below is imported statically
from `lib/` and loads its weights from the `model/` directory that carries the
same name.

## Layer pipeline

```
script.py
  |
  +- base_ensemble            LightGBM + RandomForest + CatBoost + joint state-mode
  |    +- strict_overlay          + strict_asof x 10%        (R_CORE)
  |         +- futures_fm_overlay     + FM x eta 0.10        (F)
  |              +- conditional_overlay   + regular-league FM + conditional
  |                                         correction, source-stability gated
  |
  +- form_context_rf          recent form / count context / platoon RF   -> 15%
  +- c3_sign_all              per-pitcher situational contrast consensus -> 50%
```

`base_ensemble` through `conditional_overlay` form a single chain: each layer
takes the previous layer's probability and applies one correction.  The result
of that chain is the `parent` term below.  `form_context_rf` and `c3_sign_all`
are independent components blended on top of it.

## Final blend

Applied to the R_CORE domain only (regular-season rows not involving the anchor
team); every other row passes through as `parent`:

```
clip(0.85 * parent + 0.15 * form_context_rf + 0.5 * c3, 0.001, 0.999)
```

## Module to weights

| module | weights |
|---|---|
| `lib/base_ensemble.py` | `model/base_ensemble/` |
| `lib/strict_asof.py` | `model/strict_asof/` |
| `lib/strict_overlay.py` | (blends the two above) |
| `lib/futures_fm_overlay.py` | `model/futures_fm/` |
| `lib/conditional_overlay.py` | `model/regular_fm/`, `model/conditional/` |
| `lib/row_local_features.py` | (feature construction only) |
| `lib/form_context_rf.py` | `model/form_context_rf/rf.pkl` |
| `script.py` | `model/c3_sign_all.joblib` |

## Domain codes

`R` is the KBO regular season (1군) and `F` is the Futures league (2군).
`R_ANCHOR` marks regular-season rows involving the anchor team; `R_CORE` is the
remaining regular-season population.  These are official domain codes, not
version labels.

## Row independence

Every prediction uses only the current row plus frozen training artifacts.  No
aggregate, frequency, ordering, or distribution of the evaluation set is read.
Shuffling the input rows or splitting them across calls leaves each prediction
unchanged.
"""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    found = text.count(old)
    if found != 1:
        raise ValueError(f"{label}: expected exactly 1 anchor, found {found}")
    return text.replace(old, new)


def _strip_tail(text: str, marker: str, label: str) -> str:
    found = text.count(marker)
    if found != 1:
        raise ValueError(f"{label}: expected exactly 1 tail marker, found {found}")
    return text[: text.index(marker)].rstrip("\n") + "\n"


def _inject_import(text: str, statement: str, label: str) -> str:
    return _replace_once(
        text,
        "import pandas as pd\n",
        f"import pandas as pd\n\n{statement}\n",
        f"{label}:import",
    )


def transform_base_ensemble(text: str) -> str:
    text = _strip_tail(text, "\ndef main() -> None:", "base_ensemble")
    text = _replace_once(
        text,
        '"""Offline evaluation entry point.\n'
        "\n"
        "Reads ./data/test.csv relative to this file and writes\n"
        "./output/submission.csv. Every feature is row-local or uses training artifacts\n"
        "inside ./model; no test-set aggregate, frequency, order, or distribution is\n"
        "used.\n"
        '"""\n',
        '"""Base ensemble: LightGBM, RandomForest, CatBoost and joint state-mode parts.\n'
        "\n"
        "This is the foundation of the prediction chain; the overlay modules refine\n"
        "its output.  Every feature is row-local or uses frozen training artifacts\n"
        "under ``model/base_ensemble``; no evaluation-set aggregate, frequency,\n"
        "order, or distribution is used.\n"
        '"""\n',
        "base_ensemble:docstring",
    )
    text = _replace_once(
        text,
        'BASE_DIR = Path(__file__).resolve().parent\n'
        'MODEL_DIR = BASE_DIR / "model"\n'
        'TEST_PATH = BASE_DIR / "data" / "test.csv"\n'
        'OUTPUT_PATH = BASE_DIR / "output" / "submission.csv"\n',
        'MODEL_DIR = MODEL_ROOT / "base_ensemble"\n',
        "base_ensemble:paths",
    )
    return _inject_import(text, "from lib.paths import MODEL_ROOT", "base_ensemble")


def transform_strict_asof(text: str) -> str:
    text = _strip_tail(text, "\ndef main() -> None:", "strict_asof")
    text = _replace_once(
        text,
        '"""EXP-021 final candidate inference (copied to the ZIP root as script.py)."""\n',
        '"""Strict as-of control model: group, team and low-rank pitcher effects.\n'
        "\n"
        "Every temporal feature is built from strictly as-of counters, so no value\n"
        "can depend on the outcome of the row being predicted.  Consumed by\n"
        "``strict_overlay``.\n"
        '"""\n',
        "strict_asof:docstring",
    )
    text = _replace_once(
        text,
        'MODEL_DIR = Path("./model")\n'
        'TEST_PATH = Path("./data/test.csv")\n'
        'SAMPLE_PATH = Path("./data/sample_submission.csv")\n'
        'OUTPUT_PATH = Path("./output/submission.csv")\n',
        'MODEL_DIR = MODEL_ROOT / "strict_asof"\n',
        "strict_asof:paths",
    )
    return _inject_import(text, "from lib.paths import MODEL_ROOT", "strict_asof")


def transform_strict_overlay(text: str) -> str:
    text = _strip_tail(text, "\ndef main() -> None:", "strict_overlay")
    text = _replace_once(
        text,
        '"""Standalone v82 wrapper: champion plus 10% EXP-021 strict on R_CORE."""\n',
        '"""Applies the strict as-of model as a 10% overlay on the R_CORE domain."""\n',
        "strict_overlay:docstring",
    )
    text = _replace_once(
        text,
        'BASE_DIR = Path(__file__).resolve().parent\n'
        'MODEL_DIR = BASE_DIR / "model"\n'
        'TEST_PATH = BASE_DIR / "data" / "test.csv"\n'
        'SAMPLE_PATH = BASE_DIR / "data" / "sample_submission.csv"\n'
        'OUTPUT_PATH = BASE_DIR / "output" / "submission.csv"\n',
        "",
        "strict_overlay:paths",
    )
    text = _replace_once(
        text,
        "def _load_component(name: str):\n"
        '    path = MODEL_DIR / "components" / f"{name}_script.py"\n'
        '    spec = importlib.util.spec_from_file_location(f"v82_{name}_component", path)\n'
        "    if spec is None or spec.loader is None:\n"
        '        raise ImportError(f"cannot load v82 component: {path}")\n'
        "    module = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(module)\n"
        "    module.MODEL_DIR = MODEL_DIR / name\n"
        "    return module\n\n\n",
        "",
        "strict_overlay:loader",
    )
    text = _replace_once(
        text,
        '    champion = _load_component("champion")\n'
        '    strict = _load_component("strict")\n'
        "    parent = np.asarray(champion.predict_dataframe(frame), dtype=np.float64)\n"
        "    challenger = _strict_predict(frame, strict)\n",
        "    parent = np.asarray(base_ensemble.predict_dataframe(frame), dtype=np.float64)\n"
        "    challenger = _strict_predict(frame, strict_asof)\n",
        "strict_overlay:predict",
    )
    text = _replace_once(text, "import importlib.util\n", "", "strict_overlay:importlib")
    return _inject_import(
        text, "from lib import base_ensemble, strict_asof", "strict_overlay"
    )


def transform_futures_fm_overlay(text: str) -> str:
    text = _strip_tail(text, "\ndef main() -> None:", "futures_fm_overlay")
    text = _replace_once(
        text,
        '"""Standalone v84 wrapper: Public-1159 parent plus fixed v56 FM on F."""\n',
        '"""Applies a frozen factorization-machine correction (eta 0.10) on the\n'
        "Futures (F) domain.\n"
        '"""\n',
        "futures_fm_overlay:docstring",
    )
    text = _replace_once(
        text,
        'BASE_DIR = Path(__file__).resolve().parent\n'
        'MODEL_DIR = BASE_DIR / "model"\n'
        'TEST_PATH = BASE_DIR / "data" / "test.csv"\n'
        'SAMPLE_PATH = BASE_DIR / "data" / "sample_submission.csv"\n'
        'OUTPUT_PATH = BASE_DIR / "output" / "submission.csv"\n',
        "",
        "futures_fm_overlay:paths",
    )
    text = _replace_once(
        text,
        "def _load_parent():\n"
        '    path = MODEL_DIR / "components" / "parent_script.py"\n'
        '    spec = importlib.util.spec_from_file_location("v84_parent_component", path)\n'
        "    if spec is None or spec.loader is None:\n"
        '        raise ImportError(f"cannot load v84 parent component: {path}")\n'
        "    module = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(module)\n"
        '    module.MODEL_DIR = MODEL_DIR / "parent"\n'
        "    return module\n\n\n",
        "",
        "futures_fm_overlay:loader",
    )
    text = _replace_once(
        text,
        "    parent_module = _load_parent()\n"
        "    parent = np.asarray(parent_module.predict_dataframe(frame), dtype=np.float64)\n"
        '    return apply_fixed_v56(parent, frame, MODEL_DIR / "v56_fm")\n',
        "    parent = np.asarray(strict_overlay.predict_dataframe(frame), dtype=np.float64)\n"
        '    return apply_fixed_v56(parent, frame, MODEL_ROOT / "futures_fm")\n',
        "futures_fm_overlay:predict",
    )
    text = _replace_once(text, "import importlib.util\n", "", "futures_fm_overlay:importlib")
    return _inject_import(
        text,
        "from lib import strict_overlay\nfrom lib.paths import MODEL_ROOT",
        "futures_fm_overlay",
    )


def transform_conditional_overlay(text: str) -> str:
    text = _strip_tail(text, "\ndef main() -> None:", "conditional_overlay")
    text = _replace_once(
        text,
        '"""Standalone v104 probe above the frozen Public-1161 v84 package."""\n',
        '"""Applies regular-league FM and conditional corrections, gated by a\n'
        "source-stability mask.\n"
        '"""\n',
        "conditional_overlay:docstring",
    )
    text = _replace_once(
        text,
        'BASE_DIR = Path(__file__).resolve().parent\n'
        'MODEL_DIR = BASE_DIR / "model"\n'
        'TEST_PATH = BASE_DIR / "data" / "test.csv"\n'
        'SAMPLE_PATH = BASE_DIR / "data" / "sample_submission.csv"\n'
        'OUTPUT_PATH = BASE_DIR / "output" / "submission.csv"\n',
        "",
        "conditional_overlay:paths",
    )
    text = _replace_once(
        text,
        "def _load_component(name: str, filename: str):\n"
        '    path = MODEL_DIR / "components" / filename\n'
        "    spec = importlib.util.spec_from_file_location(name, path)\n"
        "    if spec is None or spec.loader is None:\n"
        '        raise ImportError(f"cannot load component: {path}")\n'
        "    module = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(module)\n"
        "    return module\n\n\n"
        "def _load_parent():\n"
        '    module = _load_component("v104_parent_component", "parent_script.py")\n'
        '    module.MODEL_DIR = MODEL_DIR / "parent"\n'
        "    return module\n\n\n",
        "",
        "conditional_overlay:loader",
    )
    text = _replace_once(
        text,
        '    root = MODEL_DIR / "r_fm"\n',
        '    root = MODEL_ROOT / "regular_fm"\n',
        "conditional_overlay:regular_fm",
    )
    text = _replace_once(
        text,
        '    component = _load_component("v104_feature_component", "v104_features.py")\n'
        '    root = MODEL_DIR / "conditional"\n',
        '    root = MODEL_ROOT / "conditional"\n',
        "conditional_overlay:conditional",
    )
    text = _replace_once(
        text,
        "    conditional = component.feature_frame(frame, bank)\n"
        "    baseline = conditional.drop(columns=list(component.CONDITIONAL_COLUMNS))\n"
        '    conditional = component.apply_model_spec(conditional, spec["conditional"])\n'
        '    baseline = component.apply_model_spec(baseline, spec["baseline"])\n',
        "    conditional = row_local_features.feature_frame(frame, bank)\n"
        "    baseline = conditional.drop(columns=list(row_local_features.CONDITIONAL_COLUMNS))\n"
        '    conditional = row_local_features.apply_model_spec(conditional, spec["conditional"])\n'
        '    baseline = row_local_features.apply_model_spec(baseline, spec["baseline"])\n',
        "conditional_overlay:features",
    )
    text = _replace_once(
        text,
        "    parent = np.asarray(_load_parent().predict_dataframe(frame), dtype=np.float64)\n",
        "    parent = np.asarray(futures_fm_overlay.predict_dataframe(frame), dtype=np.float64)\n",
        "conditional_overlay:predict",
    )
    text = _replace_once(text, "import importlib.util\n", "", "conditional_overlay:importlib")
    return _inject_import(
        text,
        "from lib import futures_fm_overlay, row_local_features\n"
        "from lib.paths import MODEL_ROOT",
        "conditional_overlay",
    )


def transform_form_context_rf(text: str) -> str:
    text = _strip_tail(text, "\ndef main():", "form_context_rf")
    return _replace_once(
        text,
        "# script.py\n",
        '"""Recent-form, count-context and platoon random forest.\n'
        "\n"
        "An independent component blended on top of the overlay chain rather than\n"
        "a link in it.\n"
        '"""\n',
        "form_context_rf:docstring",
    )


def transform_row_local_features(text: str) -> str:
    return _replace_once(
        text,
        '"""Standalone row-local feature component for the v104 public probe."""\n',
        '"""Row-local feature construction.\n'
        "\n"
        "Every feature uses only the current row and frozen training artifacts, so\n"
        "predictions never depend on any other evaluation row.\n"
        '"""\n',
        "row_local_features:docstring",
    )


SOURCE_MAP: tuple[tuple[str, str, Any], ...] = (
    (
        "model/v124/model/parent/parent/components/champion_script.py",
        "lib/base_ensemble.py",
        transform_base_ensemble,
    ),
    (
        "model/v124/model/parent/parent/components/strict_script.py",
        "lib/strict_asof.py",
        transform_strict_asof,
    ),
    (
        "model/v124/model/parent/components/parent_script.py",
        "lib/strict_overlay.py",
        transform_strict_overlay,
    ),
    (
        "model/v124/model/components/parent_script.py",
        "lib/futures_fm_overlay.py",
        transform_futures_fm_overlay,
    ),
    (
        "model/v124/script.py",
        "lib/conditional_overlay.py",
        transform_conditional_overlay,
    ),
    ("model/h1/script.py", "lib/form_context_rf.py", transform_form_context_rf),
    (
        "model/v124/model/components/v104_features.py",
        "lib/row_local_features.py",
        transform_row_local_features,
    ),
)


def build(original_zip: Path, runtime_script: Path, output_dir: Path) -> dict[str, Any]:
    if _sha256_file(original_zip) != ORIGINAL_SHA256:
        raise ValueError("original v148 package SHA-256 mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = output_dir / "_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    artifacts: list[dict[str, str]] = []
    sources: list[dict[str, Any]] = []

    with zipfile.ZipFile(original_zip) as archive:
        members = set(archive.namelist())

        # 1. model artifacts, byte-for-byte
        mapped: set[str] = set()
        for source_prefix, destination_prefix in MODEL_MAP:
            if source_prefix.endswith("/"):
                matched = sorted(n for n in members if n.startswith(source_prefix))
                if not matched:
                    raise ValueError(f"no members under {source_prefix}")
                for name in matched:
                    relative = name[len(source_prefix) :]
                    destination = stage / (destination_prefix + relative)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    payload = archive.read(name)
                    destination.write_bytes(payload)
                    digest = _sha256_bytes(payload)
                    if _sha256_file(destination) != digest:
                        raise ValueError(f"artifact copy mismatch: {name}")
                    artifacts.append(
                        {
                            "source": name,
                            "destination": destination_prefix + relative,
                            "sha256": digest,
                        }
                    )
                    mapped.add(name)
            else:
                payload = archive.read(source_prefix)
                destination = stage / destination_prefix
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
                digest = _sha256_bytes(payload)
                if _sha256_file(destination) != digest:
                    raise ValueError(f"artifact copy mismatch: {source_prefix}")
                artifacts.append(
                    {
                        "source": source_prefix,
                        "destination": destination_prefix,
                        "sha256": digest,
                    }
                )
                mapped.add(source_prefix)

        # 2. every original member must be either mapped or deliberately dropped
        unaccounted = members - mapped - set(DROPPED)
        if unaccounted:
            raise ValueError(f"unmapped original members: {sorted(unaccounted)}")

        # 3. python components, surgically transformed
        (stage / "lib").mkdir(parents=True, exist_ok=True)
        (stage / "lib" / "__init__.py").write_text(INIT_MODULE, encoding="utf-8", newline="\n")
        (stage / "lib" / "paths.py").write_text(PATHS_MODULE, encoding="utf-8", newline="\n")
        for source_name, destination_name, transform in SOURCE_MAP:
            raw = archive.read(source_name).decode("utf-8")
            # The original package mixes CRLF (champion/strict/h1) and LF
            # (v124/v84/v82).  Normalise before anchoring; line endings in
            # Python source cannot affect a prediction.
            text = raw.replace("\r\n", "\n")
            converted = transform(text) if transform is not None else text
            (stage / destination_name).write_text(
                converted, encoding="utf-8", newline="\n"
            )
            sources.append(
                {
                    "source": source_name,
                    "destination": destination_name,
                    "original_newline": "CRLF" if "\r\n" in raw else "LF",
                    "original_lines": text.count("\n"),
                    "flat_lines": converted.count("\n"),
                    "verbatim": transform is None,
                }
            )

        # 4. entry point and the effective requirements file, verbatim
        requirements = archive.read("requirements.txt")
        (stage / "requirements.txt").write_bytes(requirements)

    # 5. layer map, so a reader never has to infer the pipeline from imports
    (stage / "ARCHITECTURE.md").write_text(
        ARCHITECTURE_DOC, encoding="utf-8", newline="\n"
    )

    shutil.copyfile(runtime_script, stage / "script.py")

    package = output_dir / "submit_v148_flat.zip"
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(stage)).replace("\\", "/"))

    return {
        "protocol": PROTOCOL,
        "package": {
            "path": str(package),
            "bytes": package.stat().st_size,
            "sha256": _sha256_file(package),
            "file_count": len(files),
        },
        "original": {
            "path": str(original_zip),
            "sha256": ORIGINAL_SHA256,
            "file_count": len(members),
        },
        "artifacts": artifacts,
        "sources": sources,
        "dropped": list(DROPPED),
        "stage": str(stage),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-zip", type=Path, required=True)
    parser.add_argument("--runtime-script", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.original_zip, args.runtime_script, args.output_dir)
    summary = {key: value for key, value in result.items() if key not in {"artifacts", "sources"}}
    summary["artifact_count"] = len(result["artifacts"])
    summary["source_count"] = len(result["sources"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
