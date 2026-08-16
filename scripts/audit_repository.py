"""Fail when the Git payload contains forbidden artifacts or likely secrets."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


ALLOWED_SPECIAL_FILES = {
    ".env.example",
    "artifacts/README.md",
    "data/README.md",
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
    if normalized in ALLOWED_SPECIAL_FILES:
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


def audit(include_untracked: bool, max_bytes: int) -> dict[str, object]:
    paths = git_paths(include_untracked)
    forbidden: list[dict[str, str]] = []
    oversized: list[dict[str, object]] = []
    secrets: list[dict[str, object]] = []
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
        if size > max_bytes:
            oversized.append({"path": path_string, "bytes": size})
        secrets.extend(secret_findings(path_string, path.read_bytes()))
    return {
        "checked_files": checked,
        "include_untracked": include_untracked,
        "max_bytes": max_bytes,
        "forbidden_paths": forbidden,
        "oversized_files": oversized,
        "secret_findings": secrets,
        "passed": not forbidden and not oversized and not secrets,
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
