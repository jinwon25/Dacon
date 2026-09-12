from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from agent_service.compliance import (
    audit_forecast_availability,
    latest_safe_forecast_run,
)
from experiments.fetch_kma_um_global import (
    DOCUMENTATION_URL,
    HISTORICAL_NOTICE_URL,
    SERIES_ENDPOINT,
    VARIABLES,
    decode_response,
    load_api_key,
)


@dataclass(frozen=True)
class ContextRequestSpec:
    reference_kst: pd.Timestamp
    initialization_utc: pd.Timestamp
    public_availability_utc: pd.Timestamp
    lead_start: int
    lead_end: int
    lead_step: int
    latitude: float
    longitude: float
    cycle_rank: int
    group: str
    nwp: str
    data_kind: str
    levels_hpa: tuple[float, ...] = ()

    @property
    def lead_hours(self) -> tuple[int, ...]:
        return tuple(range(self.lead_start, self.lead_end + 1, self.lead_step))

    @property
    def stem(self) -> str:
        cycle = self.initialization_utc.strftime("%Y%m%d%H")
        layer = "sfc" if not self.levels_hpa else "p" + "-".join(
            f"{level:g}" for level in self.levels_hpa
        )
        return (
            f"{self.group.lower()}_{self.nwp.lower()}_{cycle}_r{self.cycle_rank}_"
            f"{layer}_f{self.lead_start:03d}-{self.lead_end:03d}x{self.lead_step}"
        )


def _utc(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    return result.tz_convert("UTC")


def _kst(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("Asia/Seoul")
    return result.tz_convert("Asia/Seoul")


def context_request_specs(
    metadata: pd.DataFrame,
    *,
    latitude: float,
    longitude: float,
    group: str = "UMGL",
    nwp: str = "N128",
    cycle_count: int = 2,
    pressure_levels_hpa: tuple[float, ...] = (850.0, 700.0),
    publication_delay_hours: float = 12.0,
) -> tuple[list[ContextRequestSpec], pd.DataFrame]:
    required = {"forecast_kst_dtm", "data_available_kst_dtm"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"forecast metadata is missing columns: {sorted(missing)}")
    if cycle_count < 1:
        raise ValueError("cycle_count must be at least one")
    if publication_delay_hours < 0:
        raise ValueError("publication delay must not be negative")
    if any(level <= 0 for level in pressure_levels_hpa):
        raise ValueError("pressure levels must be positive")

    frame = metadata[list(required)].drop_duplicates().copy()
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    frame["data_available_kst_dtm"] = pd.to_datetime(
        frame["data_available_kst_dtm"]
    )
    delay = pd.Timedelta(hours=publication_delay_hours)
    specs: list[ContextRequestSpec] = []
    audit_rows: list[dict[str, object]] = []
    for reference, issue in frame.groupby("data_available_kst_dtm", sort=True):
        safe = latest_safe_forecast_run(
            reference,
            cycle_hours_utc=(0, 6, 12, 18),
            conservative_publication_delay=timedelta(
                hours=publication_delay_hours
            ),
        )
        reference_kst = _kst(reference)
        target_utc = (
            issue["forecast_kst_dtm"].dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
        )
        for cycle_rank in range(cycle_count):
            initialization = _utc(safe.initialization_utc) - pd.Timedelta(
                hours=6 * cycle_rank
            )
            availability = initialization + delay
            if availability > reference_kst.tz_convert("UTC"):
                raise ValueError("KMA context cycle is not public by the reference time")
            raw_leads = (
                target_utc - initialization
            ).dt.total_seconds().to_numpy() / 3600.0
            lead_start = int(np.floor(raw_leads.min() / 3.0) * 3)
            lead_end = int(np.ceil(raw_leads.max() / 3.0) * 3)
            if lead_start < 0:
                raise ValueError("KMA context request would require a negative lead")
            common = dict(
                reference_kst=reference_kst,
                initialization_utc=initialization,
                public_availability_utc=availability,
                lead_start=lead_start,
                lead_end=lead_end,
                lead_step=3,
                latitude=latitude,
                longitude=longitude,
                cycle_rank=cycle_rank,
                group=group,
                nwp=nwp,
            )
            specs.append(ContextRequestSpec(**common, data_kind="U"))
            if pressure_levels_hpa:
                specs.append(
                    ContextRequestSpec(
                        **common,
                        data_kind="P",
                        levels_hpa=tuple(pressure_levels_hpa),
                    )
                )
            audit_rows.append(
                {
                    "prediction_reference_kst": reference_kst,
                    "initialization_utc": initialization,
                    "public_availability_utc": availability,
                }
            )
    unique = {spec.stem: spec for spec in specs}
    if len(unique) != len(specs):
        raise ValueError("KMA context request stems are not unique")
    return list(unique.values()), pd.DataFrame(audit_rows).drop_duplicates()


def build_context_url(
    spec: ContextRequestSpec, api_key: str, *, redact: bool = False
) -> str:
    key = "<redacted>" if redact else api_key
    query: dict[str, str] = {
        "group": spec.group,
        "nwp": spec.nwp,
        "data": spec.data_kind,
        "varn": ",".join(str(value) for value in VARIABLES),
        "tmfc": spec.initialization_utc.strftime("%Y%m%d%H"),
        "ef": f"{spec.lead_start},{spec.lead_end},{spec.lead_step}",
        "lon": f"{spec.longitude:.5f}",
        "lat": f"{spec.latitude:.5f}",
        "help": "0",
        "authKey": key,
    }
    if spec.levels_hpa:
        query["level"] = ",".join(f"{level:g}" for level in spec.levels_hpa)
    return f"{SERIES_ENDPOINT}?{urllib.parse.urlencode(query)}"


def parse_context_response(text: str) -> pd.DataFrame:
    if not text.strip():
        raise ValueError("KMA API returned an empty context response")
    lower = text.lower()
    if any(token in lower for token in ("invalid auth", "error")):
        raise ValueError("KMA API returned an authentication or service error")
    pattern = re.compile(
        r"^\s*(?P<tmfc>\d{10})\s+(?P<tmef>\d{10})\s+"
        r"(?P<varn>\d+)\s+(?P<level>[-+]?\d+(?:\.\d+)?)\s+"
        r"(?P<value>[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s*$"
    )
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        variable = int(match.group("varn"))
        if variable not in VARIABLES:
            continue
        raw_level = float(match.group("level"))
        level_hpa = raw_level / 100.0 if raw_level > 2_000 else raw_level
        rows.append(
            {
                "initialization_utc": pd.to_datetime(
                    match.group("tmfc"), format="%Y%m%d%H", utc=True
                ),
                "valid_utc": pd.to_datetime(
                    match.group("tmef"), format="%Y%m%d%H", utc=True
                ),
                "variable": variable,
                "level_hpa": level_hpa,
                "value": float(match.group("value")),
            }
        )
    if not rows:
        raise ValueError("KMA context response did not contain parseable wind rows")
    result = pd.DataFrame(rows)
    keys = ["initialization_utc", "valid_utc", "variable", "level_hpa"]
    if result.duplicated(keys).any():
        raise ValueError("KMA context response contains duplicate wind rows")
    return result.sort_values(keys).reset_index(drop=True)


def validate_context_response(
    parsed: pd.DataFrame, spec: ContextRequestSpec
) -> None:
    if set(parsed["initialization_utc"]) != {spec.initialization_utc}:
        raise ValueError("KMA context initialization does not match the request")
    expected_valid = {
        spec.initialization_utc + pd.Timedelta(hours=lead)
        for lead in spec.lead_hours
    }
    if set(parsed["valid_utc"]) != expected_valid:
        raise ValueError("KMA context valid times do not match the request")
    if set(parsed["variable"]) != set(VARIABLES):
        raise ValueError("KMA context response is missing a wind component")
    if spec.levels_hpa:
        actual = {round(float(value), 4) for value in parsed["level_hpa"]}
        expected = {round(float(value), 4) for value in spec.levels_hpa}
        if actual != expected:
            raise ValueError(
                f"KMA context pressure levels differ: expected {sorted(expected)}, "
                f"received {sorted(actual)}"
            )
        expected_rows = len(expected_valid) * len(VARIABLES) * len(expected)
    else:
        if parsed.groupby(["valid_utc", "variable"]).size().ne(1).any():
            raise ValueError("KMA surface response has multiple levels per wind component")
        expected_rows = len(expected_valid) * len(VARIABLES)
    if len(parsed) != expected_rows:
        raise ValueError("KMA context response is incomplete")


def _download_context(
    spec: ContextRequestSpec, api_key: str, raw_dir: Path, retries: int
) -> dict[str, object]:
    output = raw_dir / f"{spec.stem}.txt"
    sidecar = raw_dir / f"{spec.stem}.source.json"
    if output.is_file() and sidecar.is_file():
        retained = json.loads(sidecar.read_text(encoding="utf-8"))
        if hashlib.sha256(output.read_bytes()).hexdigest() == retained.get("sha256"):
            parsed = parse_context_response(decode_response(output.read_bytes()))
            validate_context_response(parsed, spec)
            return retained
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                build_context_url(spec, api_key),
                headers={"User-Agent": "BARAM-Competition-Scientist/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            parsed = parse_context_response(decode_response(payload))
            validate_context_response(parsed, spec)
            temporary = output.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(output)
            record: dict[str, object] = {
                "path": output.as_posix(),
                "source_url": build_context_url(spec, api_key, redact=True),
                "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "prediction_reference_kst": spec.reference_kst.isoformat(),
                "initialization_utc": spec.initialization_utc.isoformat(),
                "conservative_public_availability_utc": (
                    spec.public_availability_utc.isoformat()
                ),
                "lead_hours": list(spec.lead_hours),
                "latitude": spec.latitude,
                "longitude": spec.longitude,
                "cycle_rank": spec.cycle_rank,
                "group": spec.group,
                "nwp": spec.nwp,
                "data_kind": spec.data_kind,
                "levels_hpa": list(spec.levels_hpa),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            sidecar.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return record
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"KMA context download failed for {spec.stem}: {last_error}")


def download_context_all(
    specs: list[ContextRequestSpec],
    api_key: str,
    raw_dir: Path,
    retries: int,
    workers: int,
) -> list[dict[str, object]]:
    if workers < 1:
        raise ValueError("workers must be at least one")
    if workers == 1:
        return [_download_context(spec, api_key, raw_dir, retries) for spec in specs]
    records: list[dict[str, object]] = []
    total = len(specs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_context, spec, api_key, raw_dir, retries): spec
            for spec in specs
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            records.append(future.result())
            if completed == total or completed % 25 == 0:
                print(f"KMA UM context: {completed:,}/{total:,}", flush=True)
    return sorted(records, key=lambda record: str(record["path"]))


def _record_wide(record: dict[str, object]) -> pd.DataFrame:
    parsed = parse_context_response(
        decode_response(Path(str(record["path"])).read_bytes())
    )
    kind = str(record["data_kind"])
    if kind == "U":
        if parsed.groupby(["valid_utc", "variable"]).size().ne(1).any():
            raise ValueError("KMA context surface record is ambiguous")
        wide = parsed.pivot(index="valid_utc", columns="variable", values="value")
        wide = wide.rename(columns={2002: "u10", 2003: "v10"})
    elif kind == "P":
        wide = parsed.pivot(
            index="valid_utc", columns=["variable", "level_hpa"], values="value"
        )
        wide.columns = [
            ("u" if variable == 2002 else "v") + f"{level:g}"
            for variable, level in wide.columns
        ]
    else:
        raise ValueError(f"unsupported KMA context data kind: {kind}")
    wide = wide.reset_index()
    wide["prediction_reference_kst"] = pd.Timestamp(
        str(record["prediction_reference_kst"])
    )
    wide["initialization_utc"] = pd.Timestamp(str(record["initialization_utc"]))
    wide["public_availability_utc"] = pd.Timestamp(
        str(record["conservative_public_availability_utc"])
    )
    wide["cycle_rank"] = int(record["cycle_rank"])
    return wide


def _direction_cosine(u1: pd.Series, v1: pd.Series, u2: pd.Series, v2: pd.Series) -> pd.Series:
    denominator = np.maximum(np.hypot(u1, v1) * np.hypot(u2, v2), 1e-6)
    return ((u1 * u2 + v1 * v2) / denominator).clip(-1.0, 1.0)


def build_context_features(
    metadata: pd.DataFrame,
    raw_records: list[dict[str, object]],
    output_path: Path,
) -> pd.DataFrame:
    if not raw_records:
        raise ValueError("KMA context raw records are empty")
    decoded = pd.concat([_record_wide(record) for record in raw_records], ignore_index=True)
    keys = [
        "prediction_reference_kst",
        "initialization_utc",
        "public_availability_utc",
        "cycle_rank",
        "valid_utc",
    ]
    value_columns = [column for column in decoded.columns if column not in keys]
    source = decoded.groupby(keys, as_index=False)[value_columns].first()
    required_vectors = sorted(
        {column[1:] for column in value_columns if re.fullmatch(r"[uv]\d+(?:\.\d+)?", column)}
    )
    for level in required_vectors:
        if f"u{level}" not in source or f"v{level}" not in source:
            raise ValueError(f"KMA context vector is incomplete at level {level}")

    issue_frame = metadata[
        ["forecast_kst_dtm", "data_available_kst_dtm"]
    ].drop_duplicates().copy()
    issue_frame["forecast_kst_dtm"] = pd.to_datetime(issue_frame["forecast_kst_dtm"])
    issue_frame["data_available_kst_dtm"] = pd.to_datetime(
        issue_frame["data_available_kst_dtm"]
    )
    rows: list[dict[str, object]] = []
    ranks = sorted(int(value) for value in source["cycle_rank"].unique())
    for reference, targets in issue_frame.groupby("data_available_kst_dtm", sort=True):
        reference_kst = _kst(reference)
        issue_source = source[source["prediction_reference_kst"] == reference_kst]
        if set(issue_source["cycle_rank"]) != set(ranks):
            raise ValueError(f"KMA context cycles are incomplete at {reference}")
        target_utc = (
            targets["forecast_kst_dtm"].dt.tz_localize("Asia/Seoul").dt.tz_convert("UTC")
        )
        target_x = target_utc.astype("int64").to_numpy(dtype=float)
        issue_values: dict[int, dict[str, np.ndarray | str]] = {}
        for rank in ranks:
            cycle = issue_source[issue_source["cycle_rank"] == rank].sort_values(
                "valid_utc"
            )
            if cycle["initialization_utc"].nunique() != 1:
                raise ValueError(f"KMA context rank {rank} has multiple cycles")
            x = cycle["valid_utc"].astype("int64").to_numpy(dtype=float)
            if target_x.min() < x.min() or target_x.max() > x.max():
                raise ValueError(f"KMA context rank {rank} does not bracket every target")
            values: dict[str, np.ndarray | str] = {
                "initialization": cycle["initialization_utc"].iloc[0].isoformat(),
                "availability": cycle["public_availability_utc"].iloc[0].isoformat(),
            }
            for column in value_columns:
                if cycle[column].isna().any():
                    raise ValueError(f"KMA context column {column} is incomplete")
                values[column] = np.interp(target_x, x, cycle[column].to_numpy(float))
            issue_values[rank] = values
        for position, (_, target) in enumerate(targets.iterrows()):
            row: dict[str, object] = {
                "forecast_kst_dtm": target["forecast_kst_dtm"],
                "data_available_kst_dtm": reference,
            }
            for rank, values in issue_values.items():
                row[f"initialization_utc_r{rank}"] = values["initialization"]
                row[f"public_availability_utc_r{rank}"] = values["availability"]
                for column in value_columns:
                    row[f"kma_um_ctx_{column}_r{rank}"] = float(values[column][position])
            rows.append(row)
    result = pd.DataFrame(rows).sort_values("forecast_kst_dtm").reset_index(drop=True)

    for rank in ranks:
        for level in required_vectors:
            u = result[f"kma_um_ctx_u{level}_r{rank}"]
            v = result[f"kma_um_ctx_v{level}_r{rank}"]
            result[f"kma_um_ctx_speed{level}_r{rank}"] = np.hypot(u, v)
        if "10" in required_vectors:
            for level in (item for item in required_vectors if item != "10"):
                u10 = result[f"kma_um_ctx_u10_r{rank}"]
                v10 = result[f"kma_um_ctx_v10_r{rank}"]
                upper_u = result[f"kma_um_ctx_u{level}_r{rank}"]
                upper_v = result[f"kma_um_ctx_v{level}_r{rank}"]
                result[f"kma_um_ctx_shear10_{level}_r{rank}"] = np.hypot(
                    upper_u - u10, upper_v - v10
                )
                result[f"kma_um_ctx_cos10_{level}_r{rank}"] = _direction_cosine(
                    u10, v10, upper_u, upper_v
                )
    if {0, 1}.issubset(ranks):
        for level in required_vectors:
            for component in ("u", "v", "speed"):
                result[f"kma_um_ctx_run_delta_{component}{level}"] = (
                    result[f"kma_um_ctx_{component}{level}_r0"]
                    - result[f"kma_um_ctx_{component}{level}_r1"]
                )
            result[f"kma_um_ctx_run_change{level}"] = np.hypot(
                result[f"kma_um_ctx_run_delta_u{level}"],
                result[f"kma_um_ctx_run_delta_v{level}"],
            )
            result[f"kma_um_ctx_run_cos{level}"] = _direction_cosine(
                result[f"kma_um_ctx_u{level}_r0"],
                result[f"kma_um_ctx_v{level}_r0"],
                result[f"kma_um_ctx_u{level}_r1"],
                result[f"kma_um_ctx_v{level}_r1"],
            )
    if len(result) != len(issue_frame) or result.isna().any().any():
        raise ValueError("KMA context hourly feature build is incomplete")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def _parse_levels(value: str) -> tuple[float, ...]:
    levels = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(set(levels)) != len(levels):
        raise ValueError("pressure levels must be unique")
    return levels


def replace_targets_with_issue_offset(
    metadata: pd.DataFrame,
    offset_hours: float | None,
) -> pd.DataFrame:
    """Optionally request one diagnostic target at a fixed issue-time offset.

    Negative offsets are useful for causal model-output-statistics features:
    they retrieve the archived NWP value valid at a safely lagged observation
    time, while keeping the original prediction reference as the availability
    boundary.
    """
    if offset_hours is None:
        return metadata.copy()
    required = {"forecast_kst_dtm", "data_available_kst_dtm"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"forecast metadata is missing columns: {sorted(missing)}")
    output = metadata[["data_available_kst_dtm"]].drop_duplicates().copy()
    output["forecast_kst_dtm"] = pd.to_datetime(
        output["data_available_kst_dtm"]
    ) + pd.Timedelta(hours=float(offset_hours))
    return output[["forecast_kst_dtm", "data_available_kst_dtm"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--output-dir", default="artifacts_final/external_weather/kma_um_context_2024"
    )
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--latitude", type=float, default=37.28)
    parser.add_argument("--longitude", type=float, default=128.96)
    parser.add_argument("--group", choices=("UMGL", "UMRG", "UMKR"), default="UMGL")
    parser.add_argument("--nwp", default="N128")
    parser.add_argument("--cycle-count", type=int, default=2)
    parser.add_argument("--pressure-levels", default="850,700")
    parser.add_argument("--publication-delay-hours", type=float, default=12.0)
    parser.add_argument(
        "--target-offset-hours",
        type=float,
        help=(
            "replace forecast targets with one timestamp at this offset from "
            "each issue; e.g. -2 for observation-matched MOS diagnostics"
        ),
    )
    parser.add_argument("--api-key-env", default="KMA_API_KEY")
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--max-issues", type=int)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    api_key = load_api_key(args.api_key_env, Path(args.env_file))
    if not api_key:
        raise RuntimeError(
            f"Set {args.api_key_env} or place it in {args.env_file}; keys are never "
            "accepted as command-line arguments or written to artifacts"
        )
    metadata = pd.read_csv(
        args.metadata,
        encoding="utf-8-sig",
        usecols=["forecast_kst_dtm", "data_available_kst_dtm"],
    ).drop_duplicates()
    metadata["forecast_kst_dtm"] = pd.to_datetime(metadata["forecast_kst_dtm"])
    metadata["data_available_kst_dtm"] = pd.to_datetime(
        metadata["data_available_kst_dtm"]
    )
    metadata = replace_targets_with_issue_offset(
        metadata,
        args.target_offset_hours,
    )
    metadata = metadata[
        (metadata["forecast_kst_dtm"] >= pd.Timestamp(args.start))
        & (metadata["forecast_kst_dtm"] < pd.Timestamp(args.end))
    ]
    if args.max_issues is not None:
        references = (
            metadata["data_available_kst_dtm"].drop_duplicates().sort_values().head(args.max_issues)
        )
        metadata = metadata[metadata["data_available_kst_dtm"].isin(references)]
    if metadata.empty:
        raise ValueError("No forecast metadata rows remain after filtering")
    levels = _parse_levels(args.pressure_levels)
    specs, audit_frame = context_request_specs(
        metadata,
        latitude=args.latitude,
        longitude=args.longitude,
        group=args.group,
        nwp=args.nwp,
        cycle_count=args.cycle_count,
        pressure_levels_hpa=levels,
        publication_delay_hours=args.publication_delay_hours,
    )
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = download_context_all(
        specs, api_key, raw_dir, args.retries, args.workers
    )
    features = build_context_features(metadata, records, output_dir / "features.csv")
    audit = audit_forecast_availability(audit_frame)
    manifest = {
        "schema_version": 1,
        "competition_eligible": True,
        "provider": "Korea Meteorological Administration",
        "dataset": f"KMA operational UM {args.group} {args.nwp} context archive",
        "source_type": "operational_forecast_archive",
        "documentation_url": DOCUMENTATION_URL,
        "license": "KMA public data/API terms; source attribution required",
        "license_url": "https://apihub.kma.go.kr/apiInfo.do",
        "retrieved_at_utc": pd.Timestamp.now(tz=timezone.utc).isoformat(),
        "coverage": {
            "forecast_metadata": args.metadata,
            "issues": int(metadata["data_available_kst_dtm"].nunique()),
            "targets": int(metadata["forecast_kst_dtm"].nunique()),
            "objects": len(records),
            "model": f"{args.group} {args.nwp}",
            "cycle_count": args.cycle_count,
            "pressure_levels_hpa": list(levels),
            "variables": list(VARIABLES.values()),
            "historical_access_notice": HISTORICAL_NOTICE_URL,
        },
        "availability_evidence": {
            "method": "provider_schedule_with_conservative_lag",
            "conservative_delay_minutes": int(
                round(args.publication_delay_hours * 60)
            ),
            "evidence_url": DOCUMENTATION_URL,
        },
        "causality_audit": audit,
        "feature_file": {
            "path": (output_dir / "features.csv").as_posix(),
            "rows": int(len(features)),
            "sha256": hashlib.sha256(
                (output_dir / "features.csv").read_bytes()
            ).hexdigest(),
        },
        "raw_files": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "manifest": manifest_path.as_posix(),
                "coverage": manifest["coverage"],
                "causality_audit": audit,
                "feature_file": manifest["feature_file"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
