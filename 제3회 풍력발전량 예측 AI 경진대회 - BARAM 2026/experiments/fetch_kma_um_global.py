from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
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


ENDPOINT = (
    "https://apihub.kma.go.kr/api/typ06/cgi-bin/url/"
    "nph-um_grib_pt_txt1"
)
SERIES_ENDPOINT = (
    "https://apihub.kma.go.kr/api/typ06/url/um_grib_pt_tmfc.php"
)
DOCUMENTATION_URL = "https://apihub.kma.go.kr/apiList.do?seqApi=9"
HISTORICAL_NOTICE_URL = "https://apihub.kma.go.kr/notice.do?seqNotice=52"
VARIABLES = {2002: "kma_um_u10", 2003: "kma_um_v10"}
TAGGED_VALUE = re.compile(
    r"VARN\s*[=:]\s*(?P<varn>\d+).*?VALUS?\s*[=:]\s*"
    r"(?P<value>[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)",
    re.IGNORECASE,
)


def load_api_key(environment_name: str, env_file: Path | None = None) -> str:
    """Read a KMA key from the process first, then a local dotenv file.

    The value is deliberately never accepted as a command-line argument.  This
    helper only returns it in memory and callers continue to retain redacted
    source URLs.
    """
    value = os.environ.get(environment_name, "").strip()
    if value:
        return value
    if env_file is None or not env_file.is_file():
        return ""
    for raw_line in env_file.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, candidate = line.split("=", 1)
        if name.strip() != environment_name:
            continue
        candidate = candidate.strip()
        if (
            len(candidate) >= 2
            and candidate[0] == candidate[-1]
            and candidate[0] in {"'", '"'}
        ):
            candidate = candidate[1:-1]
        return candidate.strip()
    return ""


def decode_response(payload: bytes) -> str:
    """Decode APIHub text without silently replacing malformed bytes.

    APIHub currently returns this legacy UM endpoint as CP949 even though other
    endpoints use UTF-8.  Retained raw responses stay byte-for-byte unchanged;
    only parsing uses the first strict codec that succeeds.
    """
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("KMA API response is neither valid UTF-8 nor CP949 text")


@dataclass(frozen=True)
class RequestSpec:
    reference_kst: pd.Timestamp
    initialization_utc: pd.Timestamp
    public_availability_utc: pd.Timestamp
    lead_hour: int
    latitude: float
    longitude: float
    point_id: int

    @property
    def valid_utc(self) -> pd.Timestamp:
        return self.initialization_utc + pd.Timedelta(hours=self.lead_hour)

    @property
    def stem(self) -> str:
        cycle = self.initialization_utc.strftime("%Y%m%d%H")
        return f"umgl_{cycle}_f{self.lead_hour:03d}_p{self.point_id:02d}"


@dataclass(frozen=True)
class SeriesRequestSpec:
    reference_kst: pd.Timestamp
    initialization_utc: pd.Timestamp
    public_availability_utc: pd.Timestamp
    lead_start: int
    lead_end: int
    lead_step: int
    latitude: float
    longitude: float
    point_id: int

    @property
    def lead_hours(self) -> tuple[int, ...]:
        return tuple(range(self.lead_start, self.lead_end + 1, self.lead_step))

    @property
    def stem(self) -> str:
        cycle = self.initialization_utc.strftime("%Y%m%d%H")
        return (
            f"umgl_{cycle}_f{self.lead_start:03d}-{self.lead_end:03d}"
            f"x{self.lead_step}_p{self.point_id:02d}"
        )


def _timestamp_utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _timestamp_kst(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Seoul")
    return timestamp.tz_convert("Asia/Seoul")


def request_specs(
    metadata: pd.DataFrame,
    points: tuple[tuple[float, float], ...],
    publication_delay_hours: float = 12.0,
) -> tuple[list[RequestSpec], pd.DataFrame]:
    required = {"forecast_kst_dtm", "data_available_kst_dtm"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"forecast metadata is missing columns: {sorted(missing)}")
    frame = metadata[list(required)].drop_duplicates().copy()
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    frame["data_available_kst_dtm"] = pd.to_datetime(
        frame["data_available_kst_dtm"]
    )
    specs: list[RequestSpec] = []
    audit_rows = []
    for reference, issue in frame.groupby("data_available_kst_dtm", sort=True):
        safe = latest_safe_forecast_run(
            reference,
            cycle_hours_utc=(0, 6, 12, 18),
            conservative_publication_delay=timedelta(
                hours=publication_delay_hours
            ),
        )
        initialization = _timestamp_utc(safe.initialization_utc)
        availability = _timestamp_utc(safe.conservative_publication_utc)
        valid_utc = (
            issue["forecast_kst_dtm"]
            .dt.tz_localize("Asia/Seoul")
            .dt.tz_convert("UTC")
        )
        raw_leads = (
            valid_utc - initialization
        ).dt.total_seconds().to_numpy() / 3600.0
        query_leads = set()
        for lead in raw_leads:
            query_leads.add(int(np.floor(lead / 3.0) * 3))
            query_leads.add(int(np.ceil(lead / 3.0) * 3))
        if min(query_leads) < 0:
            raise ValueError("KMA UM request would require a negative lead hour")
        for point_id, (latitude, longitude) in enumerate(points, start=1):
            for lead in sorted(query_leads):
                specs.append(
                    RequestSpec(
                        pd.Timestamp(reference),
                        initialization,
                        availability,
                        lead,
                        latitude,
                        longitude,
                        point_id,
                    )
                )
                audit_rows.append(
                    {
                        "prediction_reference_kst": reference,
                        "initialization_utc": initialization,
                        "public_availability_utc": availability,
                    }
                )
    unique = {spec.stem: spec for spec in specs}
    return list(unique.values()), pd.DataFrame(audit_rows).drop_duplicates()


def series_request_specs(
    metadata: pd.DataFrame,
    points: tuple[tuple[float, float], ...],
    publication_delay_hours: float = 12.0,
) -> tuple[list[SeriesRequestSpec], pd.DataFrame]:
    """Build one publication-time series request per issue and point."""
    required = {"forecast_kst_dtm", "data_available_kst_dtm"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"forecast metadata is missing columns: {sorted(missing)}")
    frame = metadata[list(required)].drop_duplicates().copy()
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    frame["data_available_kst_dtm"] = pd.to_datetime(
        frame["data_available_kst_dtm"]
    )
    specs: list[SeriesRequestSpec] = []
    audit_rows = []
    for reference, issue in frame.groupby("data_available_kst_dtm", sort=True):
        safe = latest_safe_forecast_run(
            reference,
            cycle_hours_utc=(0, 6, 12, 18),
            conservative_publication_delay=timedelta(
                hours=publication_delay_hours
            ),
        )
        initialization = _timestamp_utc(safe.initialization_utc)
        availability = _timestamp_utc(safe.conservative_publication_utc)
        valid_utc = (
            issue["forecast_kst_dtm"]
            .dt.tz_localize("Asia/Seoul")
            .dt.tz_convert("UTC")
        )
        raw_leads = (
            valid_utc - initialization
        ).dt.total_seconds().to_numpy() / 3600.0
        lead_start = int(np.floor(raw_leads.min() / 3.0) * 3)
        lead_end = int(np.ceil(raw_leads.max() / 3.0) * 3)
        if lead_start < 0:
            raise ValueError("KMA UM request would require a negative lead hour")
        for point_id, (latitude, longitude) in enumerate(points, start=1):
            specs.append(
                SeriesRequestSpec(
                    pd.Timestamp(reference),
                    initialization,
                    availability,
                    lead_start,
                    lead_end,
                    3,
                    latitude,
                    longitude,
                    point_id,
                )
            )
        audit_rows.append(
            {
                "prediction_reference_kst": reference,
                "initialization_utc": initialization,
                "public_availability_utc": availability,
            }
        )
    unique = {spec.stem: spec for spec in specs}
    return list(unique.values()), pd.DataFrame(audit_rows).drop_duplicates()


def build_url(spec: RequestSpec, api_key: str, *, redact: bool = False) -> str:
    key = "<redacted>" if redact else api_key
    query = urllib.parse.urlencode(
        {
            "group": "UMGL",
            "nwp": "N128",
            "data": "U",
            "varn": ",".join(str(value) for value in VARIABLES),
            "tmfc": spec.initialization_utc.strftime("%Y%m%d%H"),
            "hf": str(spec.lead_hour),
            "lon": f"{spec.longitude:.5f}",
            "lat": f"{spec.latitude:.5f}",
            "disp": "A",
            "help": "0",
            "authKey": key,
        }
    )
    return f"{ENDPOINT}?{query}"


def build_series_url(
    spec: SeriesRequestSpec, api_key: str, *, redact: bool = False
) -> str:
    key = "<redacted>" if redact else api_key
    query = urllib.parse.urlencode(
        {
            "group": "UMGL",
            "nwp": "N128",
            "data": "U",
            "varn": ",".join(str(value) for value in VARIABLES),
            "tmfc": spec.initialization_utc.strftime("%Y%m%d%H"),
            "ef": f"{spec.lead_start},{spec.lead_end},{spec.lead_step}",
            "lon": f"{spec.longitude:.5f}",
            "lat": f"{spec.latitude:.5f}",
            "help": "0",
            "authKey": key,
        }
    )
    return f"{SERIES_ENDPOINT}?{query}"


def parse_response(text: str) -> dict[int, float]:
    if not text.strip():
        raise ValueError("KMA API returned an empty response")
    lower = text.lower()
    if any(token in lower for token in ("invalid auth", "인증키", "error", "오류")):
        raise ValueError("KMA API returned an authentication or service error")
    values: dict[int, float] = {}
    for match in TAGGED_VALUE.finditer(text):
        variable = int(match.group("varn"))
        if variable in VARIABLES:
            values[variable] = float(match.group("value"))
    if len(values) == len(VARIABLES):
        return values

    # APIHub ASCII output may be a whitespace/comma table headed by TMFC/TMEF.
    for line in text.splitlines():
        tokens = re.split(r"[\s,|]+", line.strip())
        if len(tokens) < 2:
            continue
        for position, token in enumerate(tokens[:-1]):
            if token.isdigit() and int(token) in VARIABLES:
                for value_token in reversed(tokens[position + 1 :]):
                    try:
                        values[int(token)] = float(value_token)
                        break
                    except ValueError:
                        continue
    if len(values) != len(VARIABLES):
        raise ValueError(
            f"KMA response did not contain both 10 m wind components: {sorted(values)}"
        )
    return values


def parse_series_response(text: str) -> pd.DataFrame:
    """Parse a publication-time series into one row per valid time."""
    if not text.strip():
        raise ValueError("KMA API returned an empty response")
    lower = text.lower()
    if any(token in lower for token in ("invalid auth", "인증키", "error", "오류")):
        raise ValueError("KMA API returned an authentication or service error")
    rows: list[dict[str, object]] = []
    pattern = re.compile(
        r"^\s*(?P<tmfc>\d{10})\s+(?P<tmef>\d{10})\s+"
        r"(?P<varn>\d+)\s+(?P<level>[-+]?\d+(?:\.\d+)?)\s+"
        r"(?P<value>[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s*$"
    )
    for line in text.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        variable = int(match.group("varn"))
        if variable not in VARIABLES:
            continue
        rows.append(
            {
                "initialization_utc": pd.to_datetime(
                    match.group("tmfc"), format="%Y%m%d%H", utc=True
                ),
                "valid_utc": pd.to_datetime(
                    match.group("tmef"), format="%Y%m%d%H", utc=True
                ),
                "variable": variable,
                "level": float(match.group("level")),
                "value": float(match.group("value")),
            }
        )
    if not rows:
        raise ValueError("KMA series response did not contain parseable wind rows")
    long = pd.DataFrame(rows)
    if long.duplicated(["initialization_utc", "valid_utc", "variable"]).any():
        raise ValueError("KMA series response contains duplicate wind rows")
    wide = long.pivot(
        index=["initialization_utc", "valid_utc"],
        columns="variable",
        values="value",
    ).reset_index()
    missing = set(VARIABLES).difference(wide.columns)
    if missing or wide[list(VARIABLES)].isna().any().any():
        raise ValueError(
            f"KMA series response did not contain both wind components: {sorted(missing)}"
        )
    return wide.sort_values("valid_utc").reset_index(drop=True)


def _download(spec: RequestSpec, api_key: str, raw_dir: Path, retries: int) -> dict:
    output = raw_dir / f"{spec.stem}.txt"
    sidecar = raw_dir / f"{spec.stem}.source.json"
    if output.is_file() and sidecar.is_file():
        retained = json.loads(sidecar.read_text(encoding="utf-8"))
        if hashlib.sha256(output.read_bytes()).hexdigest() == retained.get("sha256"):
            return retained
    url = build_url(spec, api_key)
    redacted_url = build_url(spec, api_key, redact=True)
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "BARAM-Competition-Scientist/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            parse_response(decode_response(payload))
            temporary = output.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(output)
            retrieved = pd.Timestamp.now(tz="UTC").isoformat()
            record = {
                "path": output.as_posix(),
                "source_url": redacted_url,
                "retrieved_at_utc": retrieved,
                "initialization_utc": spec.initialization_utc.isoformat(),
                "conservative_public_availability_utc": (
                    spec.public_availability_utc.isoformat()
                ),
                "prediction_reference_kst": str(spec.reference_kst),
                "valid_utc": spec.valid_utc.isoformat(),
                "lead_hour": spec.lead_hour,
                "latitude": spec.latitude,
                "longitude": spec.longitude,
                "point_id": spec.point_id,
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
    raise RuntimeError(f"KMA download failed for {spec.stem}: {last_error}")


def _download_series(
    spec: SeriesRequestSpec, api_key: str, raw_dir: Path, retries: int
) -> dict:
    output = raw_dir / f"{spec.stem}.txt"
    sidecar = raw_dir / f"{spec.stem}.source.json"
    if output.is_file() and sidecar.is_file():
        retained = json.loads(sidecar.read_text(encoding="utf-8"))
        if hashlib.sha256(output.read_bytes()).hexdigest() == retained.get("sha256"):
            return retained
    url = build_series_url(spec, api_key)
    redacted_url = build_series_url(spec, api_key, redact=True)
    expected_valid = {
        spec.initialization_utc + pd.Timedelta(hours=lead)
        for lead in spec.lead_hours
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "BARAM-Competition-Scientist/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            parsed = parse_series_response(decode_response(payload))
            actual_valid = set(pd.DatetimeIndex(parsed["valid_utc"]))
            if actual_valid != expected_valid:
                raise ValueError(
                    "KMA series response valid times do not match the requested range"
                )
            temporary = output.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(output)
            retrieved = pd.Timestamp.now(tz="UTC").isoformat()
            record = {
                "path": output.as_posix(),
                "source_url": redacted_url,
                "retrieved_at_utc": retrieved,
                "initialization_utc": spec.initialization_utc.isoformat(),
                "conservative_public_availability_utc": (
                    spec.public_availability_utc.isoformat()
                ),
                "prediction_reference_kst": str(spec.reference_kst),
                "lead_hours": list(spec.lead_hours),
                "latitude": spec.latitude,
                "longitude": spec.longitude,
                "point_id": spec.point_id,
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
    raise RuntimeError(f"KMA series download failed for {spec.stem}: {last_error}")


def download_all(
    specs: list[RequestSpec],
    api_key: str,
    raw_dir: Path,
    retries: int,
    workers: int,
) -> list[dict]:
    """Download distinct objects with bounded concurrency and resumable files."""
    if workers < 1:
        raise ValueError("workers must be at least one")
    if workers == 1:
        return [_download(spec, api_key, raw_dir, retries) for spec in specs]

    records: list[dict] = []
    total = len(specs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download, spec, api_key, raw_dir, retries): spec
            for spec in specs
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            records.append(future.result())
            if completed == total or completed % 100 == 0:
                print(f"KMA UM objects: {completed:,}/{total:,}", flush=True)
    return sorted(records, key=lambda record: str(record["path"]))


def download_series_all(
    specs: list[SeriesRequestSpec],
    api_key: str,
    raw_dir: Path,
    retries: int,
    workers: int,
) -> list[dict]:
    if workers < 1:
        raise ValueError("workers must be at least one")
    if workers == 1:
        return [_download_series(spec, api_key, raw_dir, retries) for spec in specs]
    records: list[dict] = []
    total = len(specs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_series, spec, api_key, raw_dir, retries): spec
            for spec in specs
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            records.append(future.result())
            if completed == total or completed % 25 == 0:
                print(f"KMA UM series: {completed:,}/{total:,}", flush=True)
    return sorted(records, key=lambda record: str(record["path"]))


def build_features(
    metadata: pd.DataFrame,
    raw_records: list[dict],
    output_path: Path,
) -> pd.DataFrame:
    decoded = []
    for record in raw_records:
        values = parse_response(decode_response(Path(record["path"]).read_bytes()))
        decoded.append(
            {
                "initialization_utc": pd.Timestamp(record["initialization_utc"]),
                "valid_utc": pd.Timestamp(record["valid_utc"]),
                "point_id": int(record["point_id"]),
                "latitude": float(record["latitude"]),
                "longitude": float(record["longitude"]),
                "public_availability_utc": pd.Timestamp(
                    record["conservative_public_availability_utc"]
                ),
                **{VARIABLES[key]: value for key, value in values.items()},
            }
        )
    source = pd.DataFrame(decoded)
    return _build_features_from_source(metadata, source, output_path)


def build_series_features(
    metadata: pd.DataFrame,
    raw_records: list[dict],
    output_path: Path,
) -> pd.DataFrame:
    decoded: list[dict[str, object]] = []
    for record in raw_records:
        values = parse_series_response(
            decode_response(Path(record["path"]).read_bytes())
        )
        expected_initialization = pd.Timestamp(record["initialization_utc"])
        if set(values["initialization_utc"]) != {expected_initialization}:
            raise ValueError("KMA series initialization does not match its sidecar")
        for _, value in values.iterrows():
            decoded.append(
                {
                    "initialization_utc": value["initialization_utc"],
                    "valid_utc": value["valid_utc"],
                    "point_id": int(record["point_id"]),
                    "latitude": float(record["latitude"]),
                    "longitude": float(record["longitude"]),
                    "public_availability_utc": pd.Timestamp(
                        record["conservative_public_availability_utc"]
                    ),
                    **{
                        VARIABLES[key]: float(value[key])
                        for key in VARIABLES
                    },
                }
            )
    return _build_features_from_source(metadata, pd.DataFrame(decoded), output_path)


def _build_features_from_source(
    metadata: pd.DataFrame,
    source: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    if source.empty:
        raise ValueError("KMA UM decoded source is empty")
    rows = []
    issue_frame = metadata[
        ["forecast_kst_dtm", "data_available_kst_dtm"]
    ].drop_duplicates()
    issue_frame["forecast_kst_dtm"] = pd.to_datetime(
        issue_frame["forecast_kst_dtm"]
    )
    issue_frame["data_available_kst_dtm"] = pd.to_datetime(
        issue_frame["data_available_kst_dtm"]
    )
    for reference, targets in issue_frame.groupby("data_available_kst_dtm", sort=True):
        reference_utc = _timestamp_kst(reference).tz_convert("UTC")
        issue_source = source[
            source["public_availability_utc"] <= reference_utc
        ]
        if issue_source.empty:
            raise ValueError(
                f"No KMA UM cycle was public by prediction reference {reference}"
            )
        initialization = issue_source["initialization_utc"].max()
        issue_source = issue_source[issue_source["initialization_utc"] == initialization]
        if issue_source["point_id"].nunique() != source["point_id"].nunique():
            raise ValueError(
                f"Latest KMA UM cycle is incomplete across points at {reference}"
            )
        for point_id, point in issue_source.groupby("point_id"):
            point = point.sort_values("valid_utc")
            x = point["valid_utc"].astype("int64").to_numpy(dtype=float)
            target_utc = (
                targets["forecast_kst_dtm"]
                .dt.tz_localize("Asia/Seoul")
                .dt.tz_convert("UTC")
            )
            target_x = target_utc.astype("int64").to_numpy(dtype=float)
            if target_x.min() < x.min() or target_x.max() > x.max():
                raise ValueError(
                    f"KMA UM lead range does not bracket every target for point {point_id}"
                )
            for target_position, (_, target) in enumerate(targets.iterrows()):
                u10 = float(np.interp(target_x[target_position], x, point["kma_um_u10"]))
                v10 = float(np.interp(target_x[target_position], x, point["kma_um_v10"]))
                rows.append(
                    {
                        "forecast_kst_dtm": target["forecast_kst_dtm"],
                        "data_available_kst_dtm": reference,
                        "point_id": int(point_id),
                        "latitude": float(point["latitude"].iloc[0]),
                        "longitude": float(point["longitude"].iloc[0]),
                        "initialization_utc": initialization.isoformat(),
                        "public_availability_utc": point[
                            "public_availability_utc"
                        ].iloc[0].isoformat(),
                        "kma_um_u10": u10,
                        "kma_um_v10": v10,
                        "kma_um_speed10": float(np.hypot(u10, v10)),
                    }
                )
    result = pd.DataFrame(rows).sort_values(
        ["forecast_kst_dtm", "point_id"]
    )
    expected = len(issue_frame) * source["point_id"].nunique()
    if len(result) != expected or result.isna().any().any():
        raise ValueError("KMA UM hourly feature build is incomplete")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--output-dir", default="artifacts_final/external_weather/kma_um_global_2024"
    )
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--latitude", type=float, default=37.28)
    parser.add_argument("--longitude", type=float, default=128.96)
    parser.add_argument("--publication-delay-hours", type=float, default=12.0)
    parser.add_argument("--api-key-env", default="KMA_API_KEY")
    parser.add_argument(
        "--env-file",
        default=".env.local",
        help="Local dotenv fallback; the key is never copied into artifacts.",
    )
    parser.add_argument("--max-issues", type=int)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--request-mode",
        choices=("series", "point"),
        default="series",
        help="Use one publication-time series per issue or the legacy per-lead endpoint.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Bounded concurrent API requests; retained files make reruns resumable.",
    )
    args = parser.parse_args()

    api_key = load_api_key(args.api_key_env, Path(args.env_file))
    if not api_key:
        raise RuntimeError(
            f"Set {args.api_key_env} or place it in {args.env_file}; "
            "keys are never accepted as command-line arguments or written to artifacts"
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
    metadata = metadata[
        (metadata["forecast_kst_dtm"] >= pd.Timestamp(args.start))
        & (metadata["forecast_kst_dtm"] < pd.Timestamp(args.end))
    ]
    if args.max_issues is not None:
        retained = (
            metadata["data_available_kst_dtm"]
            .drop_duplicates()
            .sort_values()
            .head(args.max_issues)
        )
        metadata = metadata[metadata["data_available_kst_dtm"].isin(retained)]
    if metadata.empty:
        raise ValueError("No forecast metadata rows remain after filtering")

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    points = ((args.latitude, args.longitude),)
    if args.request_mode == "series":
        specs, audit_frame = series_request_specs(
            metadata, points, args.publication_delay_hours
        )
        raw_records = download_series_all(
            specs, api_key, raw_dir, args.retries, args.workers
        )
        features = build_series_features(
            metadata, raw_records, output_dir / "features.csv"
        )
    else:
        specs, audit_frame = request_specs(
            metadata, points, args.publication_delay_hours
        )
        raw_records = download_all(
            specs, api_key, raw_dir, args.retries, args.workers
        )
        features = build_features(
            metadata, raw_records, output_dir / "features.csv"
        )
    audit = audit_forecast_availability(audit_frame)
    retrieved = pd.Timestamp.now(tz=timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "competition_eligible": True,
        "provider": "Korea Meteorological Administration",
        "dataset": "KMA operational UM global N128 forecast point archive",
        "source_type": "operational_forecast_archive",
        "documentation_url": DOCUMENTATION_URL,
        "license": "KMA public data/API terms; source attribution required",
        "license_url": "https://apihub.kma.go.kr/apiInfo.do",
        "retrieved_at_utc": retrieved,
        "coverage": {
            "forecast_metadata": args.metadata,
            "issues": int(metadata["data_available_kst_dtm"].nunique()),
            "targets": int(metadata["forecast_kst_dtm"].nunique()),
            "objects": len(raw_records),
            "request_mode": args.request_mode,
            "model": "UMGL N128",
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
        "raw_files": raw_records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
