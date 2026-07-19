from __future__ import annotations

import argparse
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
DOCUMENTATION_URL = "https://apihub.kma.go.kr/apiList.do?seqApi=9"
HISTORICAL_NOTICE_URL = "https://apihub.kma.go.kr/notice.do?seqNotice=52"
VARIABLES = {2002: "kma_um_u10", 2003: "kma_um_v10"}
TAGGED_VALUE = re.compile(
    r"VARN\s*[=:]\s*(?P<varn>\d+).*?VALUS?\s*[=:]\s*"
    r"(?P<value>[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)",
    re.IGNORECASE,
)


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
            parse_response(payload.decode("utf-8", errors="replace"))
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


def build_features(
    metadata: pd.DataFrame,
    raw_records: list[dict],
    output_path: Path,
) -> pd.DataFrame:
    decoded = []
    for record in raw_records:
        values = parse_response(Path(record["path"]).read_text(encoding="utf-8"))
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
    parser.add_argument("--max-issues", type=int)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Set {args.api_key_env} to a user-issued KMA APIHub key; "
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
    specs, audit_frame = request_specs(
        metadata,
        ((args.latitude, args.longitude),),
        args.publication_delay_hours,
    )
    raw_records = [
        _download(spec, api_key, raw_dir, args.retries) for spec in specs
    ]
    features = build_features(metadata, raw_records, output_dir / "features.csv")
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
