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
# byte-for-byte; only their path changes.
MODEL_MAP: tuple[tuple[str, str], ...] = (
    ("model/v124/model/parent/parent/champion/", "model/champion/"),
    ("model/v124/model/parent/parent/strict/", "model/strict/"),
    ("model/v124/model/conditional/", "model/conditional/"),
    ("model/v124/model/r_fm/", "model/r_fm/"),
    ("model/v124/model/parent/v56_fm/", "model/v56_fm/"),
    ("model/h1/model/rf.pkl", "model/h1/rf.pkl"),
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

INIT_MODULE = '"""Frozen v148 inference components (flat single-generation layout)."""\n'


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


def transform_champion(text: str) -> str:
    text = _strip_tail(text, "\ndef main() -> None:", "champion")
    text = _replace_once(
        text,
        'BASE_DIR = Path(__file__).resolve().parent\n'
        'MODEL_DIR = BASE_DIR / "model"\n'
        'TEST_PATH = BASE_DIR / "data" / "test.csv"\n'
        'OUTPUT_PATH = BASE_DIR / "output" / "submission.csv"\n',
        'MODEL_DIR = MODEL_ROOT / "champion"\n',
        "champion:paths",
    )
    return _inject_import(text, "from lib.paths import MODEL_ROOT", "champion")


def transform_strict(text: str) -> str:
    text = _strip_tail(text, "\ndef main() -> None:", "strict")
    text = _replace_once(
        text,
        'MODEL_DIR = Path("./model")\n'
        'TEST_PATH = Path("./data/test.csv")\n'
        'SAMPLE_PATH = Path("./data/sample_submission.csv")\n'
        'OUTPUT_PATH = Path("./output/submission.csv")\n',
        'MODEL_DIR = MODEL_ROOT / "strict"\n',
        "strict:paths",
    )
    return _inject_import(text, "from lib.paths import MODEL_ROOT", "strict")


def transform_v82(text: str) -> str:
    text = _strip_tail(text, "\ndef main() -> None:", "v82")
    text = _replace_once(
        text,
        'BASE_DIR = Path(__file__).resolve().parent\n'
        'MODEL_DIR = BASE_DIR / "model"\n'
        'TEST_PATH = BASE_DIR / "data" / "test.csv"\n'
        'SAMPLE_PATH = BASE_DIR / "data" / "sample_submission.csv"\n'
        'OUTPUT_PATH = BASE_DIR / "output" / "submission.csv"\n',
        "",
        "v82:paths",
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
        "v82:loader",
    )
    text = _replace_once(
        text,
        '    champion = _load_component("champion")\n'
        '    strict = _load_component("strict")\n'
        "    parent = np.asarray(champion.predict_dataframe(frame), dtype=np.float64)\n",
        "    parent = np.asarray(champion.predict_dataframe(frame), dtype=np.float64)\n",
        "v82:predict",
    )
    text = _replace_once(text, "import importlib.util\n", "", "v82:importlib")
    return _inject_import(text, "from lib import champion, strict", "v82")


def transform_v84(text: str) -> str:
    text = _strip_tail(text, "\ndef main() -> None:", "v84")
    text = _replace_once(
        text,
        'BASE_DIR = Path(__file__).resolve().parent\n'
        'MODEL_DIR = BASE_DIR / "model"\n'
        'TEST_PATH = BASE_DIR / "data" / "test.csv"\n'
        'SAMPLE_PATH = BASE_DIR / "data" / "sample_submission.csv"\n'
        'OUTPUT_PATH = BASE_DIR / "output" / "submission.csv"\n',
        "",
        "v84:paths",
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
        "v84:loader",
    )
    text = _replace_once(
        text,
        "    parent_module = _load_parent()\n"
        "    parent = np.asarray(parent_module.predict_dataframe(frame), dtype=np.float64)\n"
        '    return apply_fixed_v56(parent, frame, MODEL_DIR / "v56_fm")\n',
        "    parent = np.asarray(v82.predict_dataframe(frame), dtype=np.float64)\n"
        '    return apply_fixed_v56(parent, frame, MODEL_ROOT / "v56_fm")\n',
        "v84:predict",
    )
    text = _replace_once(text, "import importlib.util\n", "", "v84:importlib")
    return _inject_import(
        text, "from lib import v82\nfrom lib.paths import MODEL_ROOT", "v84"
    )


def transform_v124(text: str) -> str:
    text = _strip_tail(text, "\ndef main() -> None:", "v124")
    text = _replace_once(
        text,
        'BASE_DIR = Path(__file__).resolve().parent\n'
        'MODEL_DIR = BASE_DIR / "model"\n'
        'TEST_PATH = BASE_DIR / "data" / "test.csv"\n'
        'SAMPLE_PATH = BASE_DIR / "data" / "sample_submission.csv"\n'
        'OUTPUT_PATH = BASE_DIR / "output" / "submission.csv"\n',
        "",
        "v124:paths",
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
        "v124:loader",
    )
    text = _replace_once(
        text,
        '    root = MODEL_DIR / "r_fm"\n',
        '    root = MODEL_ROOT / "r_fm"\n',
        "v124:r_fm",
    )
    text = _replace_once(
        text,
        '    component = _load_component("v104_feature_component", "v104_features.py")\n'
        '    root = MODEL_DIR / "conditional"\n',
        '    root = MODEL_ROOT / "conditional"\n',
        "v124:conditional",
    )
    text = _replace_once(
        text,
        "    conditional = component.feature_frame(frame, bank)\n"
        "    baseline = conditional.drop(columns=list(component.CONDITIONAL_COLUMNS))\n"
        '    conditional = component.apply_model_spec(conditional, spec["conditional"])\n'
        '    baseline = component.apply_model_spec(baseline, spec["baseline"])\n',
        "    conditional = v104_features.feature_frame(frame, bank)\n"
        "    baseline = conditional.drop(columns=list(v104_features.CONDITIONAL_COLUMNS))\n"
        '    conditional = v104_features.apply_model_spec(conditional, spec["conditional"])\n'
        '    baseline = v104_features.apply_model_spec(baseline, spec["baseline"])\n',
        "v124:features",
    )
    text = _replace_once(
        text,
        "    parent = np.asarray(_load_parent().predict_dataframe(frame), dtype=np.float64)\n",
        "    parent = np.asarray(v84.predict_dataframe(frame), dtype=np.float64)\n",
        "v124:predict",
    )
    text = _replace_once(text, "import importlib.util\n", "", "v124:importlib")
    return _inject_import(
        text,
        "from lib import v104_features, v84\nfrom lib.paths import MODEL_ROOT",
        "v124",
    )


def transform_h1(text: str) -> str:
    return _strip_tail(text, "\ndef main():", "h1")


SOURCE_MAP: tuple[tuple[str, str, Any], ...] = (
    (
        "model/v124/model/parent/parent/components/champion_script.py",
        "lib/champion.py",
        transform_champion,
    ),
    (
        "model/v124/model/parent/parent/components/strict_script.py",
        "lib/strict.py",
        transform_strict,
    ),
    (
        "model/v124/model/parent/components/parent_script.py",
        "lib/v82.py",
        transform_v82,
    ),
    (
        "model/v124/model/components/parent_script.py",
        "lib/v84.py",
        transform_v84,
    ),
    ("model/v124/script.py", "lib/v124.py", transform_v124),
    ("model/h1/script.py", "lib/h1.py", transform_h1),
    (
        "model/v124/model/components/v104_features.py",
        "lib/v104_features.py",
        None,
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
