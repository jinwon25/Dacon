from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_service.contracts import CompetitionProfile, resolve_inside


@dataclass(frozen=True)
class ServiceConfig:
    project_root: Path
    raw: dict[str, Any]

    @property
    def database_path(self) -> Path:
        return resolve_inside(self.project_root, self.raw["database_path"], "database_path")

    @property
    def run_root(self) -> Path:
        return resolve_inside(self.project_root, self.raw["run_root"], "run_root")

    @property
    def submission_dir(self) -> Path:
        return resolve_inside(self.project_root, self.raw["submission_dir"], "submission_dir")

    @property
    def archive_dir(self) -> Path:
        return resolve_inside(self.project_root, self.raw["archive_dir"], "archive_dir")

    @property
    def policy(self) -> dict[str, Any]:
        return dict(self.raw["policy"])

    @property
    def roles(self) -> dict[str, str]:
        return {str(key): str(value) for key, value in self.raw["roles"].items()}

    @property
    def allowed_module_prefixes(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.raw["allowed_module_prefixes"])

    @property
    def human_submission_required(self) -> bool:
        return bool(self.raw.get("human_submission_required", True))

    @property
    def submission(self) -> dict[str, Any]:
        return dict(self.raw.get("submission", {}))

    @property
    def governance(self) -> dict[str, Any]:
        return dict(self.raw.get("governance", {}))

    @property
    def competition_profile(self) -> CompetitionProfile:
        relative = str(
            self.raw.get("competition_profile_path", ".agents/competition.json")
        )
        path = resolve_inside(self.project_root, relative, "competition_profile_path")
        return CompetitionProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @property
    def competition_slug(self) -> str:
        if "competition_profile_path" in self.raw:
            return self.competition_profile.slug
        return str(self.raw.get("competition_slug", "baram_2026"))


def load_config(project_root: Path, config_path: Path | None = None) -> ServiceConfig:
    root = Path(project_root).resolve()
    path = config_path or root / ".agents" / "baram.json"
    path = Path(path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Agent config must stay inside the project root") from exc
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "database_path",
        "run_root",
        "submission_dir",
        "archive_dir",
        "allowed_module_prefixes",
        "policy",
        "roles",
    }
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Agent config is missing: {sorted(missing)}")
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported agent config schema_version")
    return ServiceConfig(root, raw)
