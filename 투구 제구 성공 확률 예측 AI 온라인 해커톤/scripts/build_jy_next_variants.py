from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "jy_combo_bridge025_h1w016_20260828_01" / "submit_jy_combo_bridge025_h1w016.zip"
OUT_DIR = ROOT / "artifacts" / "jy_next_variants_20260828_01"


VARIANTS = {
    "jy_runners_high_li": {
        "zip": "submit_jy_runners_high_li.zip",
        "gate": 'runners_on | high_li',
        "bridge_scale": "1.0",
        "description": "R_CORE rows with runners on or li >= 1.5 receive the bridge025/H1/C3 update.",
    },
    "jy_runners_pressure_count": {
        "zip": "submit_jy_runners_pressure_count.zip",
        "gate": 'runners_on | pressure_count',
        "bridge_scale": "1.0",
        "description": "R_CORE rows with runners on or a pressure count receive the bridge025/H1/C3 update.",
    },
    "jy_runners_bridge022": {
        "zip": "submit_jy_runners_bridge022.zip",
        "gate": 'runners_on',
        "bridge_scale": "0.7",
        "description": "Runners-on gate only, with effective bridge strength reduced from 0.25 to about 0.22.",
    },
    "jy_runners_bridge027": {
        "zip": "submit_jy_runners_bridge027.zip",
        "gate": 'runners_on',
        "bridge_scale": "1.2",
        "description": "Runners-on gate only, with effective bridge strength increased from 0.25 to about 0.27.",
    },
    "jy_runners_high_li_bridge027": {
        "zip": "submit_jy_runners_high_li_bridge027.zip",
        "gate": 'runners_on | high_li',
        "bridge_scale": "1.2",
        "description": "Sequential final: runners-on or li >= 1.5 gate, with effective bridge strength increased to about 0.27.",
    },
}


BASE_REPLACEMENTS = {
    "H1_WEIGHT = 0.16\nC3_WEIGHT = 0.5\nMEAN_RECENT_WEIGHT = 0.25\n": (
        "H1_WEIGHT = 0.16\n"
        "H1_BASE_WEIGHT = 0.15\n"
        "C3_WEIGHT = 0.5\n"
        "MEAN_RECENT_WEIGHT = 0.25\n"
        "MEAN_RECENT_BASE_WEIGHT = 0.15\n"
        "BRIDGE_SCALE = {bridge_scale}\n"
    ),
    "def _predict_c3(frame: pd.DataFrame) -> np.ndarray:\n": (
        "def _predict_c3(frame: pd.DataFrame, recent_weight: float = MEAN_RECENT_WEIGHT) -> np.ndarray:\n"
    ),
    "    return (1.0 - MEAN_RECENT_WEIGHT) * sign_all + MEAN_RECENT_WEIGHT * mean_recent\n": (
        "    return (1.0 - recent_weight) * sign_all + recent_weight * mean_recent\n"
    ),
}


PREDICT_OLD = """    c3 = _predict_c3(frame)
    active = _active_mask(frame)
    output = parent.copy()
    output[active] = np.clip(
        (1.0 - H1_WEIGHT) * bridge_parent[active]
        + H1_WEIGHT * h1[active]
        + C3_WEIGHT * c3[active],
        0.001,
        0.999,
    )
    return parent, h1, c3, active, output
"""


PREDICT_NEW = """    c3_base = _predict_c3(frame, MEAN_RECENT_BASE_WEIGHT)
    c3 = _predict_c3(frame, MEAN_RECENT_WEIGHT)
    base_active = _active_mask(frame)
    runners_on = pd.to_numeric(frame["num_runners_on"], errors="coerce").fillna(0).to_numpy() > 0
    high_li = pd.to_numeric(frame["li"], errors="coerce").fillna(0.0).to_numpy() >= 1.5
    pressure_count = (
        (pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).to_numpy() >= 3)
        | (pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).to_numpy() >= 2)
    )
    active = base_active & ({gate})
    effective_bridge = parent + BRIDGE_SCALE * (bridge_parent - parent)
    output = parent.copy()
    output[base_active] = np.clip(
        (1.0 - H1_BASE_WEIGHT) * parent[base_active]
        + H1_BASE_WEIGHT * h1[base_active]
        + C3_WEIGHT * c3_base[base_active],
        0.001,
        0.999,
    )
    output[active] = np.clip(
        (1.0 - H1_WEIGHT) * effective_bridge[active]
        + H1_WEIGHT * h1[active]
        + C3_WEIGHT * c3[active],
        0.001,
        0.999,
    )
    return parent, h1, c3, active, output
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def build_script(template: str, variant: dict[str, str]) -> str:
    text = template
    for old, new in BASE_REPLACEMENTS.items():
        if old not in text:
            raise RuntimeError(f"replacement point not found: {old[:80]!r}")
        text = text.replace(old, new.format(**variant), 1)
    if PREDICT_OLD not in text:
        raise RuntimeError("predict_components block not found")
    return text.replace(PREDICT_OLD, PREDICT_NEW.format(**variant), 1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifests = []
    with zipfile.ZipFile(SOURCE, "r") as src:
        original_script = src.read("script.py").decode("utf-8")
        for name, variant in VARIANTS.items():
            target = OUT_DIR / variant["zip"]
            patched_script = build_script(original_script, variant)
            with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as dst:
                for info in src.infolist():
                    data = patched_script.encode("utf-8") if info.filename == "script.py" else src.read(info.filename)
                    dst.writestr(info, data)
            manifest = {
                "name": name,
                "candidate_submit": str(target.relative_to(ROOT)),
                "sha256": sha256(target),
                **variant,
            }
            manifests.append(manifest)
            print(json.dumps(manifest, ensure_ascii=False, indent=2))

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
