from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
MODULE = re.compile(r"^experiments\.[a-z][a-z0-9_]*$")


def _required_text(value: object, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _finite(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class Hypothesis:
    title: str
    family: str
    rationale: str
    expected_signal: str
    source_urls: tuple[str, ...] = field(default_factory=tuple)
    risk_notes: str = ""
    competition_slug: str = "baram_2026"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Hypothesis":
        family = _required_text(raw.get("family"), "family")
        if not IDENTIFIER.fullmatch(family):
            raise ValueError("family must be a snake_case identifier")
        sources = tuple(str(url).strip() for url in raw.get("source_urls", []) if str(url).strip())
        if any(not url.startswith(("https://", "http://")) for url in sources):
            raise ValueError("source_urls must contain HTTP(S) URLs")
        return cls(
            title=_required_text(raw.get("title"), "title"),
            family=family,
            rationale=_required_text(raw.get("rationale"), "rationale"),
            expected_signal=_required_text(raw.get("expected_signal"), "expected_signal"),
            source_urls=sources,
            risk_notes=str(raw.get("risk_notes", "")).strip(),
            competition_slug=_identifier(
                raw.get("competition_slug", "baram_2026"), "competition_slug"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["source_urls"] = list(self.source_urls)
        return output


@dataclass(frozen=True)
class RunSpec:
    hypothesis_id: int
    module: str
    args: tuple[str, ...]
    report_path: str
    evaluation_path: str
    candidate_path: str | None = None
    timeout_seconds: int = 3_600
    input_paths: tuple[str, ...] = field(default_factory=tuple)
    external_manifest_paths: tuple[str, ...] = field(default_factory=tuple)
    tags: dict[str, str] = field(default_factory=dict)
    parent_run_id: int | None = None
    validation_plan_id: int | None = None
    change_summary: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunSpec":
        module = _required_text(raw.get("module"), "module")
        if not MODULE.fullmatch(module):
            raise ValueError("module must be an experiments.<snake_case> module")
        timeout = int(raw.get("timeout_seconds", 3_600))
        if not 1 <= timeout <= 86_400:
            raise ValueError("timeout_seconds must be between 1 and 86400")
        candidate = raw.get("candidate_path")
        parent_run_id = raw.get("parent_run_id")
        validation_plan_id = raw.get("validation_plan_id")
        change_summary = str(raw.get("change_summary", "")).strip()
        if parent_run_id is not None and not change_summary:
            raise ValueError("a child run must describe its single bounded change")
        return cls(
            hypothesis_id=int(raw["hypothesis_id"]),
            module=module,
            args=tuple(str(value) for value in raw.get("args", [])),
            report_path=_required_text(raw.get("report_path"), "report_path"),
            evaluation_path=_required_text(raw.get("evaluation_path"), "evaluation_path"),
            candidate_path=str(candidate).strip() if candidate else None,
            timeout_seconds=timeout,
            input_paths=tuple(str(value) for value in raw.get("input_paths", [])),
            external_manifest_paths=tuple(
                str(value) for value in raw.get("external_manifest_paths", [])
            ),
            tags={str(key): str(value) for key, value in raw.get("tags", {}).items()},
            parent_run_id=None if parent_run_id is None else int(parent_run_id),
            validation_plan_id=(
                None if validation_plan_id is None else int(validation_plan_id)
            ),
            change_summary=change_summary,
        )

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["args"] = list(self.args)
        output["input_paths"] = list(self.input_paths)
        output["external_manifest_paths"] = list(self.external_manifest_paths)
        return output

    def materialize(self, run_id: int) -> "RunSpec":
        replacement = str(int(run_id))

        def expand(value: str | None) -> str | None:
            return None if value is None else value.replace("{run_id}", replacement)

        return RunSpec(
            hypothesis_id=self.hypothesis_id,
            module=self.module,
            args=tuple(str(expand(value)) for value in self.args),
            report_path=str(expand(self.report_path)),
            evaluation_path=str(expand(self.evaluation_path)),
            candidate_path=expand(self.candidate_path),
            timeout_seconds=self.timeout_seconds,
            input_paths=tuple(str(expand(value)) for value in self.input_paths),
            external_manifest_paths=tuple(
                str(expand(value)) for value in self.external_manifest_paths
            ),
            tags=dict(self.tags),
            parent_run_id=self.parent_run_id,
            validation_plan_id=self.validation_plan_id,
            change_summary=self.change_summary,
        )


@dataclass(frozen=True)
class Evaluation:
    family: str
    locked_score_delta: float
    locked_one_minus_nmae_delta: float
    locked_ficr_delta: float
    expected_macro_score_delta: float
    positive_months: int
    total_months: int
    bootstrap_positive_fraction: float
    bootstrap_q05: float
    changed_ratio: float
    p95_movement_ratio: float
    # Stable metadata used by the public-evidence guard.  Defaults keep old
    # evaluation reports and adapters backwards compatible.
    family_group: str | None = None
    direction: str = "unknown"
    worst_month_score_delta: float | None = None
    notes: str = ""
    fold_scores: tuple[float, ...] = field(default_factory=tuple)
    cv_mean: float | None = None
    cv_std: float | None = None
    oof_path: str | None = None
    runtime_seconds: float | None = None
    peak_memory_mb: float | None = None
    leakage_risk: str = "unknown"
    rule_violation: str = "none"
    selection_metric: float | None = None
    selection_direction: str = "maximize"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Evaluation":
        family = _required_text(raw.get("family"), "family")
        if not IDENTIFIER.fullmatch(family):
            raise ValueError("family must be a snake_case identifier")
        positive_months = int(raw["positive_months"])
        total_months = int(raw["total_months"])
        if total_months <= 0 or not 0 <= positive_months <= total_months:
            raise ValueError("positive_months must be between 0 and total_months")
        fold_scores = tuple(
            _finite(value, "fold_scores") for value in raw.get("fold_scores", [])
        )
        leakage_risk = str(raw.get("leakage_risk", "unknown")).strip().lower()
        if leakage_risk not in {"low", "medium", "high", "unknown"}:
            raise ValueError("leakage_risk must be low, medium, high, or unknown")
        family_group_raw = raw.get("family_group")
        family_group = (
            family
            if family_group_raw is None or not str(family_group_raw).strip()
            else str(family_group_raw).strip()
        )
        if not IDENTIFIER.fullmatch(family_group):
            raise ValueError("family_group must be a snake_case identifier")
        direction = str(
            raw.get("direction", raw.get("change_direction", "unknown"))
        ).strip().lower() or "unknown"
        if not IDENTIFIER.fullmatch(direction):
            raise ValueError("direction must be a snake_case identifier")
        selection_direction = str(
            raw.get("selection_direction", "maximize")
        ).strip().lower()
        if selection_direction not in {"maximize", "minimize"}:
            raise ValueError("selection_direction must be maximize or minimize")

        def optional_finite(name: str) -> float | None:
            value = raw.get(name)
            return None if value is None else _finite(value, name)

        result = cls(
            family=family,
            locked_score_delta=_finite(raw["locked_score_delta"], "locked_score_delta"),
            locked_one_minus_nmae_delta=_finite(
                raw["locked_one_minus_nmae_delta"], "locked_one_minus_nmae_delta"
            ),
            locked_ficr_delta=_finite(raw["locked_ficr_delta"], "locked_ficr_delta"),
            expected_macro_score_delta=_finite(
                raw["expected_macro_score_delta"], "expected_macro_score_delta"
            ),
            positive_months=positive_months,
            total_months=total_months,
            bootstrap_positive_fraction=_finite(
                raw["bootstrap_positive_fraction"], "bootstrap_positive_fraction"
            ),
            bootstrap_q05=_finite(raw["bootstrap_q05"], "bootstrap_q05"),
            changed_ratio=_finite(raw["changed_ratio"], "changed_ratio"),
            p95_movement_ratio=_finite(raw["p95_movement_ratio"], "p95_movement_ratio"),
            family_group=family_group,
            direction=direction,
            worst_month_score_delta=optional_finite("worst_month_score_delta"),
            notes=str(raw.get("notes", "")).strip(),
            fold_scores=fold_scores,
            cv_mean=optional_finite("cv_mean"),
            cv_std=optional_finite("cv_std"),
            oof_path=(str(raw["oof_path"]).strip() if raw.get("oof_path") else None),
            runtime_seconds=optional_finite("runtime_seconds"),
            peak_memory_mb=optional_finite("peak_memory_mb"),
            leakage_risk=leakage_risk,
            rule_violation=str(raw.get("rule_violation", "none")).strip().lower(),
            selection_metric=optional_finite("selection_metric"),
            selection_direction=selection_direction,
        )
        for name in ("bootstrap_positive_fraction", "changed_ratio", "p95_movement_ratio"):
            value = getattr(result, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        return result

    @property
    def positive_month_fraction(self) -> float:
        return self.positive_months / self.total_months

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["fold_scores"] = list(self.fold_scores)
        return output


def _identifier(value: object, name: str) -> str:
    identifier = _required_text(value, name)
    if not IDENTIFIER.fullmatch(identifier):
        raise ValueError(f"{name} must be a snake_case identifier")
    return identifier


@dataclass(frozen=True)
class CompetitionProfile:
    slug: str
    name: str
    platform: str
    competition_id: str
    task_type: str
    metric_name: str
    metric_direction: str
    sample_submission_path: str
    id_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    target_ranges: dict[str, tuple[float | None, float | None]]
    time_columns: tuple[str, ...] = field(default_factory=tuple)
    group_columns: tuple[str, ...] = field(default_factory=tuple)
    rules: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CompetitionProfile":
        direction = str(raw.get("metric_direction", "maximize")).lower()
        if direction not in {"maximize", "minimize"}:
            raise ValueError("metric_direction must be maximize or minimize")
        targets = tuple(
            _required_text(value, "target_columns")
            for value in raw.get("target_columns", [])
        )
        if not targets:
            raise ValueError("target_columns must not be empty")
        ranges: dict[str, tuple[float | None, float | None]] = {}
        for target in targets:
            bounds = raw.get("target_ranges", {}).get(target, {})
            minimum = bounds.get("min")
            maximum = bounds.get("max")
            ranges[target] = (
                None if minimum is None else _finite(minimum, f"{target}.min"),
                None if maximum is None else _finite(maximum, f"{target}.max"),
            )
        return cls(
            slug=_identifier(raw.get("slug"), "slug"),
            name=_required_text(raw.get("name"), "name"),
            platform=str(raw.get("platform", "other")).strip().lower(),
            competition_id=_required_text(raw.get("competition_id"), "competition_id"),
            task_type=_required_text(raw.get("task_type"), "task_type"),
            metric_name=_required_text(raw.get("metric_name"), "metric_name"),
            metric_direction=direction,
            sample_submission_path=_required_text(
                raw.get("sample_submission_path"), "sample_submission_path"
            ),
            id_columns=tuple(str(value) for value in raw.get("id_columns", [])),
            target_columns=targets,
            target_ranges=ranges,
            time_columns=tuple(str(value) for value in raw.get("time_columns", [])),
            group_columns=tuple(str(value) for value in raw.get("group_columns", [])),
            rules={str(key): str(value) for key, value in raw.get("rules", {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        for key in ("id_columns", "target_columns", "time_columns", "group_columns"):
            output[key] = list(output[key])
        output["target_ranges"] = {
            target: {"min": bounds[0], "max": bounds[1]}
            for target, bounds in self.target_ranges.items()
        }
        return output


@dataclass(frozen=True)
class ValidationPlan:
    competition_slug: str
    name: str
    method: str
    rationale: str
    n_splits: int
    time_column: str | None = None
    group_columns: tuple[str, ...] = field(default_factory=tuple)
    embargo_rows: int = 0
    leakage_checks: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ValidationPlan":
        n_splits = int(raw.get("n_splits", 1))
        if not 1 <= n_splits <= 100:
            raise ValueError("n_splits must be between 1 and 100")
        embargo = int(raw.get("embargo_rows", 0))
        if embargo < 0:
            raise ValueError("embargo_rows must not be negative")
        return cls(
            competition_slug=_identifier(
                raw.get("competition_slug"), "competition_slug"
            ),
            name=_required_text(raw.get("name"), "name"),
            method=_identifier(raw.get("method"), "method"),
            rationale=_required_text(raw.get("rationale"), "rationale"),
            n_splits=n_splits,
            time_column=(
                str(raw["time_column"]).strip() if raw.get("time_column") else None
            ),
            group_columns=tuple(str(value) for value in raw.get("group_columns", [])),
            embargo_rows=embargo,
            leakage_checks=tuple(str(value) for value in raw.get("leakage_checks", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["group_columns"] = list(self.group_columns)
        output["leakage_checks"] = list(self.leakage_checks)
        return output


def resolve_inside(root: Path, relative: str, name: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{name} must stay inside the project root: {relative}") from exc
    return candidate
