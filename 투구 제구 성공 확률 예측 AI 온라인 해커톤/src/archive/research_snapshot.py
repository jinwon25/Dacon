"""Create a deterministic code archive and SHA-256 reproducibility manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _files_under(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.rglob("*") if item.is_file())


def _run_git(repo_root: Path, arguments: list[str]) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={repo_root.as_posix()}",
        *arguments,
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout + (f"\nSTDERR:\n{result.stderr}" if result.stderr else "")


def _code_files(project_dir: Path) -> list[Path]:
    files: set[Path] = set()
    for name in ("README.md", ".gitignore", "requirements.txt", "data_description.md"):
        path = project_dir / name
        if path.exists():
            files.add(path)
    files.update(project_dir.glob("*.py"))
    for directory in ("src", "tests", "configs", "reports"):
        files.update(_files_under(project_dir / directory))
    return sorted(
        path
        for path in files
        if "__pycache__" not in path.parts and path.suffix not in {".pyc", ".pyo"}
        and path != project_dir / "research" / "reports" / "reproducibility_manifest.csv"
    )


def _write_deterministic_zip(
    project_dir: Path,
    output: Path,
    files: list[Path],
    extra_files: dict[str, Path],
) -> None:
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        entries = {
            path.relative_to(project_dir).as_posix(): path for path in files
        }
        entries.update(extra_files)
        for name in sorted(entries):
            path = entries[name]
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 6, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def run(project_dir: Path, snapshot_name: str) -> dict[str, object]:
    project_dir = project_dir.resolve()
    repo_root = project_dir.parent
    snapshot_dir = project_dir / "artifacts" / snapshot_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    relative_project = project_dir.relative_to(repo_root).as_posix()

    raw_git_status = _run_git(
        repo_root,
        ["status", "--short", "--untracked-files=all", "--", relative_project],
    )
    # Exclude the snapshot's own generated files so repeated snapshot runs
    # describe the same pre-existing research state.
    git_status = "\n".join(
        line
        for line in raw_git_status.splitlines()
        if snapshot_name not in line and "reproducibility_manifest.csv" not in line
    )
    if git_status:
        git_status += "\n"
    git_diff = _run_git(
        repo_root, ["diff", "--binary", "--no-ext-diff", "--", relative_project]
    )
    git_status_path = snapshot_dir / "git_status.txt"
    git_diff_path = snapshot_dir / "git_diff.patch"
    git_status_path.write_text(git_status, encoding="utf-8")
    git_diff_path.write_text(git_diff, encoding="utf-8")

    code_files = _code_files(project_dir)
    code_zip = snapshot_dir / "code_snapshot.zip"
    _write_deterministic_zip(
        project_dir,
        code_zip,
        code_files,
        {
            "_snapshot/git_status.txt": git_status_path,
            "_snapshot/git_diff.patch": git_diff_path,
        },
    )

    groups: dict[str, list[Path]] = {
        "data": _files_under(project_dir / "data"),
        "config": _files_under(project_dir / "research" / "configs"),
        "incumbent_model": _files_under(project_dir / "model"),
        "followup_model": _files_under(project_dir / "artifacts" / "followup" / "models"),
        "oof_cache": _files_under(project_dir / "artifacts" / "followup" / "oof"),
        "candidate_artifact": _files_under(
            project_dir / "artifacts" / "candidates"
        ),
        "submission": [
            path
            for path in (
                project_dir / "submit.zip",
                project_dir / "artifacts" / "incumbent" / "submission1_submit.zip",
            )
            if path.exists()
        ]
        + sorted(project_dir.glob("submit_candidate_*.zip")),
    }
    description = project_dir / "data_description.md"
    if description.exists():
        groups["data"].append(description)

    manifest_rows: list[dict[str, object]] = []
    for group, paths in groups.items():
        for path in sorted(set(paths)):
            manifest_rows.append(
                {
                    "group": group,
                    "path": path.relative_to(project_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    manifest_rows.extend(
        [
            {
                "group": "snapshot",
                "path": code_zip.relative_to(project_dir).as_posix(),
                "size_bytes": code_zip.stat().st_size,
                "sha256": _sha256(code_zip),
            },
            {
                "group": "git_state",
                "path": git_status_path.relative_to(project_dir).as_posix(),
                "size_bytes": git_status_path.stat().st_size,
                "sha256": _sha256(git_status_path),
            },
            {
                "group": "git_state",
                "path": git_diff_path.relative_to(project_dir).as_posix(),
                "size_bytes": git_diff_path.stat().st_size,
                "sha256": _sha256(git_diff_path),
            },
        ]
    )

    manifest_path = project_dir / "research" / "reports" / "reproducibility_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["group", "path", "size_bytes", "sha256"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    submission_rows = [row for row in manifest_rows if row["group"] == "submission"]
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "project_dir": str(project_dir),
        "snapshot_name": snapshot_name,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "code_snapshot_path": str(code_zip),
        "code_snapshot_sha256": _sha256(code_zip),
        "git_status_path": str(git_status_path),
        "git_diff_path": str(git_diff_path),
        "git_diff_is_empty": git_diff.strip() == "",
        "git_note": (
            "The project is untracked in the parent repository, so standard git diff "
            "is empty. git_status.txt plus the deterministic code snapshot preserve state."
        ),
        "incumbent_submission_hashes_match": (
            len(
                [
                    row
                    for row in submission_rows
                    if row["path"]
                    in {"submit.zip", "artifacts/incumbent/submission1_submit.zip"}
                ]
            )
            == 2
            and len(
                {
                    row["sha256"]
                    for row in submission_rows
                    if row["path"]
                    in {"submit.zip", "artifacts/incumbent/submission1_submit.zip"}
                }
            )
            == 1
        ),
    }
    metadata_path = snapshot_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--snapshot-name", default="research_snapshot_20260806")
    args = parser.parse_args()
    print(json.dumps(run(args.project_dir, args.snapshot_name), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
