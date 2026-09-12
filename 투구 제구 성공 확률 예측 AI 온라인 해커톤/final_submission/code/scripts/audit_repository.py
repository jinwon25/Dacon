"""Fail when the Git payload contains forbidden artifacts or likely secrets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


ALLOWED_SPECIAL_FILES = {
    ".env.example",
    "artifacts/README.md",
    "artifacts/oof_champion_1161/README.md",
    "artifacts/oof_champion_1161/manifest.json",
    "artifacts/oof_champion_1161/v84_full_2022.npz",
    "artifacts/oof_champion_1161/v84_full_2024.npz",
    "artifacts/oof_champion_1161/v84_late_2023.npz",
    "artifacts/oof_champion_1170/README.md",
    "artifacts/oof_champion_1170/manifest.json",
    "artifacts/standalone_champion_1161/standalone_champion_1161.zip",
    "artifacts/standalone_champion_1161/standalone_manifest.json",
    "artifacts/standalone_champion_1162/standalone_champion_1162.zip",
    "artifacts/standalone_champion_1162/standalone_manifest.json",
    "data/README.md",
}
ALLOWED_LFS_FILES = {
    "artifacts/oof_champion_1161/v84_full_2022.npz",
    "artifacts/oof_champion_1161/v84_full_2024.npz",
    "artifacts/oof_champion_1161/v84_late_2023.npz",
    "artifacts/standalone_champion_1161/standalone_champion_1161.zip",
    "artifacts/standalone_champion_1162/standalone_champion_1162.zip",
}
PINNED_REGULAR_BINARY_FILES = {
    "fallback_xgb/fallback_lookups.joblib": {
        "bytes": 862_763,
        "sha256": "374E7BA22AD181F488A082C04A0BBCE456B164F45D05F62FEF3AF209030DB3C5",
    },
    "artifacts/oof_champion_1170/v148_full_2024.npz": {
        "bytes": 8_168_945,
        "sha256": "972383D4DE481DCF8A27FCC497C9C88BC08FCD6612B08C4E3F0E37A824EDF2F9",
    },
    "submissions/releases/v167/submit_v167.zip": {
        "bytes": 46_354_018,
        "sha256": "30DD28F56723EC0F560C9101FC5A94EF78568F874DCA88BF808879831E61C8C1",
    },
    "submissions/releases/row_region_gate_bridge027/submit_row_region_gate_bridge027.zip": {
        "bytes": 83_675_308,
        "sha256": "4C924E046091304BFC73B50BE51110BDF1351DFD9B577738CC6B65A8DFF43C9E",
    },
    "submissions/releases/v335/submit_v335_anchor_lowrank_complement.zip": {
        "bytes": 91_469_892,
        "sha256": "A3BCD933DE3F78DEB9565F537DC3199922BDFD219B2840E4150F939CFF144C7A",
    },
}
FORBIDDEN_SUFFIXES = {
    ".cbm",
    ".ckpt",
    ".feather",
    ".joblib",
    ".key",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pem",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".zip",
}
SECRET_PATTERNS = {
    "github_token": re.compile(
        rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
    ),
    "private_key": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
    "aws_access_key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "openai_key": re.compile(rb"sk-[A-Za-z0-9_-]{30,}"),
}


def forbidden_path_reason(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    lower = normalized.lower()
    if (
        normalized in ALLOWED_SPECIAL_FILES
        or normalized in PINNED_REGULAR_BINARY_FILES
    ):
        return None
    if lower == ".env" or lower.startswith(".env."):
        return "environment credential file"
    if lower.startswith(("model/", "output/")):
        return "generated model or output directory"
    if lower.startswith("data/"):
        return "DACON-provided data"
    if lower.startswith("artifacts/"):
        return "generated model or OOF artifact"
    if Path(lower).suffix in FORBIDDEN_SUFFIXES:
        return "binary artifact or submission archive"
    return None


def secret_findings(path: str, data: bytes) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    for name, pattern in SECRET_PATTERNS.items():
        if pattern.search(data):
            findings.append({"path": path, "pattern": name})
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return findings
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = re.match(r"^\s*DACON_(?:API_)?TOKEN\s*=\s*(.*?)\s*$", line)
        if match and match.group(1).strip().strip("\"'"):
            findings.append(
                {
                    "path": path,
                    "pattern": "nonempty_dacon_token",
                    "line": line_number,
                }
            )
    return findings


def git_paths(include_untracked: bool) -> list[str]:
    command = [
        "git",
        "-c",
        f"safe.directory={REPOSITORY_ROOT.as_posix()}",
        "-C",
        str(REPOSITORY_ROOT),
        "ls-files",
        "--cached",
    ]
    if include_untracked:
        command.extend(["--others", "--exclude-standard"])
    command.append("-z")
    raw = subprocess.check_output(command)
    return sorted(
        path.decode("utf-8") for path in raw.split(b"\0") if path
    )


def lfs_filter(path: str) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={REPOSITORY_ROOT.as_posix()}",
        "-C",
        str(REPOSITORY_ROOT),
        "check-attr",
        "filter",
        "--",
        path,
    ]
    output = subprocess.check_output(command, text=True).strip()
    return output.rsplit(":", 1)[-1].strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def audit(include_untracked: bool, max_bytes: int) -> dict[str, object]:
    paths = git_paths(include_untracked)
    forbidden: list[dict[str, str]] = []
    oversized: list[dict[str, object]] = []
    secrets: list[dict[str, object]] = []
    lfs_misconfigured: list[dict[str, str]] = []
    pinned_binary_mismatches: list[dict[str, object]] = []
    checked = 0
    for path_string in paths:
        path = REPOSITORY_ROOT / path_string
        if not path.is_file():
            continue
        checked += 1
        reason = forbidden_path_reason(path_string)
        if reason:
            forbidden.append({"path": path_string, "reason": reason})
        size = path.stat().st_size
        if (
            size > max_bytes
            and path_string not in ALLOWED_LFS_FILES
            and path_string not in PINNED_REGULAR_BINARY_FILES
        ):
            oversized.append({"path": path_string, "bytes": size})
        if path_string in ALLOWED_LFS_FILES:
            if lfs_filter(path_string) != "lfs":
                lfs_misconfigured.append(
                    {"path": path_string, "reason": "missing filter=lfs"}
                )
        elif path_string in PINNED_REGULAR_BINARY_FILES:
            expected = PINNED_REGULAR_BINARY_FILES[path_string]
            actual_sha256 = sha256_file(path)
            if size != expected["bytes"] or actual_sha256 != expected["sha256"]:
                pinned_binary_mismatches.append(
                    {
                        "path": path_string,
                        "expected_bytes": expected["bytes"],
                        "actual_bytes": size,
                        "expected_sha256": expected["sha256"],
                        "actual_sha256": actual_sha256,
                    }
                )
        else:
            secrets.extend(secret_findings(path_string, path.read_bytes()))
    return {
        "checked_files": checked,
        "include_untracked": include_untracked,
        "max_bytes": max_bytes,
        "forbidden_paths": forbidden,
        "oversized_files": oversized,
        "secret_findings": secrets,
        "lfs_misconfigured": lfs_misconfigured,
        "pinned_binary_mismatches": pinned_binary_mismatches,
        "passed": not forbidden
        and not oversized
        and not secrets
        and not lfs_misconfigured
        and not pinned_binary_mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-untracked", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=5_000_000)
    args = parser.parse_args()
    result = audit(args.include_untracked, args.max_bytes)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
