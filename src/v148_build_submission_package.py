"""Build and audit the v148 conservative v142-to-v138 bridge package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.v124_public_quadratic_stack import build_package as build_intermediate
from src.v142_build_submission_package import audit_package, build_package
from src.core.packaging import _sha256


PROTOCOL = "V148_V142_V138_BLEND_PACKAGE_V1"


def run(
    parent_v104: Path,
    h1_zip: Path,
    runtime_script: Path,
    train_csv: Path,
    oof_path: Path,
    data_dir: Path,
    v124_config_path: Path,
    config_path: Path,
    output_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"config protocol must be {PROTOCOL}")
    if _sha256(parent_v104) != str(config["parent_v104_sha256"]).upper():
        raise ValueError("v104 parent SHA mismatch")
    if _sha256(h1_zip) != str(config["h1_package_sha256"]).upper():
        raise ValueError("H1 package SHA mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = json.loads(v124_config_path.read_text(encoding="utf-8"))
    base_config["package_name"] = "base_v148.zip"
    base_config["selected_point"] = config["intermediate_point"]
    base_dir = output_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)
    base_zip, base_build, base_diff = build_intermediate(parent_v104, base_config, base_dir)

    final_config = {
        "package_name": config["package_name"],
        "c3": config["c3"],
    }
    package, build = build_package(
        base_zip, h1_zip, runtime_script, train_csv, oof_path, final_config, output_dir
    )
    audit = audit_package(
        package,
        base_zip,
        data_dir,
        train_csv,
        output_dir,
        timeout,
        float(config["runtime_limit_seconds"]),
        int(config["scale_proxy_rows"]),
    )
    result = {
        "protocol": PROTOCOL,
        "status": "eligible_for_api_submission",
        "formula": (
            "0.85*intermediate(v124->v104,w=.15) + 0.15*H1 + "
            "0.5*(0.85*sign_all_C3 + 0.15*mean_recent_C3) on R_CORE"
        ),
        "package": {
            "path": str(package),
            "bytes": package.stat().st_size,
            "sha256": _sha256(package),
        },
        "intermediate_base": {
            "path": str(base_zip),
            "sha256": _sha256(base_zip),
            "point": config["intermediate_point"],
            "build": base_build,
            "archive_diff": base_diff,
        },
        "build": build,
        "audit": audit,
        "eligible_for_api_submission": True,
        **config["restrictions"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=float), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-v104", type=Path, required=True)
    parser.add_argument("--h1-zip", type=Path, required=True)
    parser.add_argument("--runtime-script", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--oof-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--v124-config", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    run(
        args.parent_v104,
        args.h1_zip,
        args.runtime_script,
        args.train_csv,
        args.oof_path,
        args.data_dir,
        args.v124_config,
        args.config,
        args.output_dir,
        args.timeout,
    )


if __name__ == "__main__":
    main()
