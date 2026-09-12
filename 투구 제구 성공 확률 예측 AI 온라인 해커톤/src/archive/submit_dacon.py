"""Guarded DACON code-submission helper for competition 236743."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path


COMPETITION_ID = "236743"


def _load_credentials(
    project_dir: Path, credentials_file: Path | None = None
) -> tuple[str | None, str | None]:
    token = os.environ.get("DACON_API_TOKEN") or os.environ.get("DACON_TOKEN")
    team = os.environ.get("DACON_TEAM_NAME") or os.environ.get("DACON_TEAM")
    env_path = credentials_file or (project_dir / ".env")
    if (not token or not team) and env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip().strip('"').strip("'")
            if key.strip() in {"DACON_API_TOKEN", "DACON_TOKEN"} and not token:
                token = value
            if key.strip() in {"DACON_TEAM_NAME", "DACON_TEAM"} and not team:
                team = value
    return token, team


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def audit(path: Path) -> dict:
    if not path.exists() or path.suffix.lower() != ".zip":
        raise ValueError("submission must be an existing ZIP file")
    if len(path.name) > 30:
        raise ValueError("DACON code-submission filename must be at most 30 characters")
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
    roots = sorted({name.split("/", 1)[0] for name in members if name})
    required = {"model", "script.py", "requirements.txt"}
    if not required.issubset(roots):
        raise ValueError(f"missing required package roots: {sorted(required - set(roots))}")
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "roots": roots,
        "competition_id": COMPETITION_ID,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--credentials-file", type=Path)
    parser.add_argument("--memo", default="")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    credentials_file = args.credentials_file
    if credentials_file and not credentials_file.is_absolute():
        credentials_file = (project_dir / credentials_file).resolve()
    path = args.zip_path if args.zip_path.is_absolute() else project_dir / args.zip_path
    result = audit(path)
    token, team = _load_credentials(project_dir, credentials_file)
    result["credentials_present"] = bool(token and team)
    result["team_name_present"] = bool(team)
    result["mode"] = "execute" if args.execute else "dry_run"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.execute:
        return
    if not token or not team:
        raise SystemExit(
            "DACON credentials are missing. Set DACON_API_TOKEN/DACON_TEAM_NAME "
            "or create a git-ignored .env with those two keys."
        )
    from dacon_submit_api import dacon_submit_api

    response = dacon_submit_api.post_code_submission_file(
        str(path), token, COMPETITION_ID, team, args.memo
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))
    if not isinstance(response, dict) or not response.get("isSubmitted"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
