from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from agent_service.compliance import audit_forecast_availability
from experiments.fetch_kma_um_global import load_api_key


ENDPOINT = (
    "https://apihub.kma.go.kr/api/typ02/openApi/"
    "VilageFcstInfoService_2.0/getVilageFcst"
)
DOCUMENTATION_URL = (
    "https://apihub.kma.go.kr/apiList.do?seqApi=10&seqApiSub=286"
)
CATEGORIES = {
    "UUU": "kma_village_u10",
    "VVV": "kma_village_v10",
    "WSD": "kma_village_speed10",
    "TMP": "kma_village_temperature",
    "REH": "kma_village_humidity",
    "SKY": "kma_village_sky",
    "PTY": "kma_village_precipitation_type",
}
CYCLE_HOURS_KST = (2, 5, 8, 11, 14, 17, 20, 23)
KEYS = ["forecast_kst_dtm", "data_available_kst_dtm"]


@dataclass(frozen=True)
class VillageRequestSpec:
    reference_kst: pd.Timestamp
    base_kst: pd.Timestamp
    public_availability_kst: pd.Timestamp
    nx: int
    ny: int

    @property
    def stem(self) -> str:
        return f"village_{self.base_kst.strftime('%Y%m%d%H')}_x{self.nx:03d}_y{self.ny:03d}"


def _kst(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("Asia/Seoul")
    return timestamp.tz_convert("Asia/Seoul")


def latest_safe_village_cycle(
    reference_kst: object, publication_delay_minutes: int = 60
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if publication_delay_minutes < 0:
        raise ValueError("publication delay must not be negative")
    reference = _kst(reference_kst)
    candidates = []
    for day_offset in range(3):
        day = reference.normalize() - pd.Timedelta(days=day_offset)
        for hour in CYCLE_HOURS_KST:
            base = day + pd.Timedelta(hours=hour)
            public = base + pd.Timedelta(minutes=publication_delay_minutes)
            if public <= reference:
                candidates.append((base, public))
    if not candidates:
        raise ValueError("no village forecast cycle is public by the reference time")
    return max(candidates, key=lambda item: item[0])


def village_request_specs(
    metadata: pd.DataFrame,
    *,
    nx: int = 94,
    ny: int = 121,
    publication_delay_minutes: int = 60,
) -> tuple[list[VillageRequestSpec], pd.DataFrame]:
    required = {"forecast_kst_dtm", "data_available_kst_dtm"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"forecast metadata is missing columns: {sorted(missing)}")
    if not 1 <= nx <= 149 or not 1 <= ny <= 253:
        raise ValueError("village forecast grid is outside the documented domain")
    references = pd.to_datetime(metadata["data_available_kst_dtm"]).drop_duplicates()
    specs = []
    audit = []
    for reference in sorted(references):
        base, public = latest_safe_village_cycle(
            reference, publication_delay_minutes
        )
        spec = VillageRequestSpec(_kst(reference), base, public, nx, ny)
        specs.append(spec)
        audit.append(
            {
                "prediction_reference_kst": spec.reference_kst,
                "initialization_utc": spec.base_kst.tz_convert("UTC"),
                "public_availability_utc": spec.public_availability_kst.tz_convert(
                    "UTC"
                ),
            }
        )
    unique = {spec.stem: spec for spec in specs}
    if len(unique) != len(specs):
        raise ValueError("village request stems are not unique")
    return list(unique.values()), pd.DataFrame(audit)


def build_url(
    spec: VillageRequestSpec, api_key: str, *, redact: bool = False
) -> str:
    query = urllib.parse.urlencode(
        {
            "pageNo": "1",
            "numOfRows": "2000",
            "dataType": "JSON",
            "base_date": spec.base_kst.strftime("%Y%m%d"),
            "base_time": spec.base_kst.strftime("%H00"),
            "nx": str(spec.nx),
            "ny": str(spec.ny),
            "authKey": "<redacted>" if redact else api_key,
        }
    )
    return f"{ENDPOINT}?{query}"


def parse_response(text: str) -> pd.DataFrame:
    if not text.strip():
        raise ValueError("KMA village API returned an empty response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("KMA village API response is not JSON") from error
    try:
        response = payload["response"]
        header = response["header"]
        body = response["body"]
    except (KeyError, TypeError) as error:
        raise ValueError("KMA village API response schema is incomplete") from error
    if str(header.get("resultCode", "")) != "00":
        raise ValueError(
            f"KMA village API error: {header.get('resultCode', 'unknown')} "
            f"{header.get('resultMsg', '')}"
        )
    items = body.get("items", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list) or not items:
        raise ValueError("KMA village API returned no forecast items")
    total = int(body.get("totalCount", len(items)))
    if total > len(items):
        raise ValueError("KMA village response was paginated and is incomplete")
    rows = []
    for item in items:
        category = str(item.get("category", ""))
        if category not in CATEGORIES:
            continue
        forecast = pd.to_datetime(
            f"{item.get('fcstDate', '')}{str(item.get('fcstTime', '')).zfill(4)}",
            format="%Y%m%d%H%M",
            errors="coerce",
        )
        try:
            value = float(item.get("fcstValue"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"KMA village {category} is not numeric") from error
        if pd.isna(forecast) or not np.isfinite(value):
            raise ValueError("KMA village forecast item is malformed")
        rows.append(
            {
                "base_kst": pd.to_datetime(
                    f"{item.get('baseDate', '')}{str(item.get('baseTime', '')).zfill(4)}",
                    format="%Y%m%d%H%M",
                    errors="raise",
                ).tz_localize("Asia/Seoul"),
                "forecast_kst_dtm": forecast,
                "category": category,
                "value": value,
                "nx": int(item.get("nx")),
                "ny": int(item.get("ny")),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("KMA village response has no retained forecast categories")
    keys = ["base_kst", "forecast_kst_dtm", "category", "nx", "ny"]
    if result.duplicated(keys).any():
        raise ValueError("KMA village response contains duplicate forecast items")
    return result.sort_values(keys).reset_index(drop=True)


def validate_response(parsed: pd.DataFrame, spec: VillageRequestSpec) -> None:
    if set(parsed["base_kst"]) != {spec.base_kst}:
        raise ValueError("KMA village response base time differs from the request")
    if set(parsed["nx"]) != {spec.nx} or set(parsed["ny"]) != {spec.ny}:
        raise ValueError("KMA village response grid differs from the request")
    wind = parsed[parsed["category"].isin(("UUU", "VVV", "WSD"))]
    counts = wind.groupby("forecast_kst_dtm")["category"].nunique()
    if wind.empty or not counts.eq(3).all():
        raise ValueError("KMA village response has incomplete wind vectors")


def _download(
    spec: VillageRequestSpec, api_key: str, raw_dir: Path, retries: int
) -> dict[str, object]:
    output = raw_dir / f"{spec.stem}.json"
    sidecar = raw_dir / f"{spec.stem}.source.json"
    if output.is_file() and sidecar.is_file():
        retained = json.loads(sidecar.read_text(encoding="utf-8"))
        if hashlib.sha256(output.read_bytes()).hexdigest() == retained.get("sha256"):
            validate_response(parse_response(output.read_text(encoding="utf-8")), spec)
            return retained
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                build_url(spec, api_key),
                headers={"User-Agent": "BARAM-Competition-Scientist/1.0"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            text = payload.decode("utf-8-sig")
            validate_response(parse_response(text), spec)
            temporary = output.with_suffix(".tmp")
            temporary.write_bytes(payload)
            temporary.replace(output)
            record: dict[str, object] = {
                "path": output.as_posix(),
                "source_url": build_url(spec, api_key, redact=True),
                "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "prediction_reference_kst": spec.reference_kst.isoformat(),
                "initialization_utc": spec.base_kst.tz_convert("UTC").isoformat(),
                "conservative_public_availability_utc": (
                    spec.public_availability_kst.tz_convert("UTC").isoformat()
                ),
                "base_kst": spec.base_kst.isoformat(),
                "nx": spec.nx,
                "ny": spec.ny,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            sidecar.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return record
        except (OSError, ValueError, UnicodeError, urllib.error.URLError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"KMA village download failed for {spec.stem}: {last_error}")


def download_all(
    specs: list[VillageRequestSpec],
    api_key: str,
    raw_dir: Path,
    retries: int,
    workers: int,
) -> list[dict[str, object]]:
    if workers < 1:
        raise ValueError("workers must be at least one")
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download, spec, api_key, raw_dir, retries): spec
            for spec in specs
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            records.append(future.result())
            if completed == len(specs) or completed % 25 == 0:
                print(f"KMA village issues: {completed:,}/{len(specs):,}", flush=True)
    return sorted(records, key=lambda record: str(record["path"]))


def build_features(
    metadata: pd.DataFrame,
    records: list[dict[str, object]],
    output_path: Path,
) -> pd.DataFrame:
    decoded = []
    for record in records:
        parsed = parse_response(Path(str(record["path"])).read_text(encoding="utf-8"))
        wide = parsed.pivot(
            index="forecast_kst_dtm", columns="category", values="value"
        ).rename(columns=CATEGORIES)
        wide["data_available_kst_dtm"] = pd.Timestamp(
            str(record["prediction_reference_kst"])
        ).tz_localize(None)
        wide["base_kst"] = str(record["base_kst"])
        wide["public_availability_utc"] = str(
            record["conservative_public_availability_utc"]
        )
        decoded.append(wide.reset_index())
    source = pd.concat(decoded, ignore_index=True)
    targets = metadata[KEYS].drop_duplicates().copy()
    for key in KEYS:
        targets[key] = pd.to_datetime(targets[key])
    result = targets.merge(source, on=KEYS, how="left", validate="one_to_one")
    required = list(CATEGORIES.values())
    if result[required].isna().any().any():
        raise ValueError("KMA village forecast does not cover every target/category")
    result["kma_village_vector_speed10"] = np.hypot(
        result["kma_village_u10"], result["kma_village_v10"]
    )
    result["kma_village_speed_vector_delta"] = (
        result["kma_village_speed10"] - result["kma_village_vector_speed10"]
    )
    result = result.sort_values("forecast_kst_dtm").reset_index(drop=True)
    if len(result) != len(targets) or result.isna().any().any():
        raise ValueError("KMA village feature build is incomplete")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/train/gfs_train.csv")
    parser.add_argument(
        "--output-dir", default="artifacts_final/external_weather/kma_village_2024"
    )
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--nx", type=int, default=94)
    parser.add_argument("--ny", type=int, default=121)
    parser.add_argument("--publication-delay-minutes", type=int, default=60)
    parser.add_argument("--api-key-env", default="KMA_API_KEY")
    parser.add_argument("--env-file", default=".env.local")
    parser.add_argument("--max-issues", type=int)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    api_key = load_api_key(args.api_key_env, Path(args.env_file))
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} or place it in {args.env_file}")
    metadata = pd.read_csv(
        args.metadata,
        encoding="utf-8-sig",
        usecols=KEYS,
    ).drop_duplicates()
    for key in KEYS:
        metadata[key] = pd.to_datetime(metadata[key])
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
    specs, audit_frame = village_request_specs(
        metadata,
        nx=args.nx,
        ny=args.ny,
        publication_delay_minutes=args.publication_delay_minutes,
    )
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = download_all(specs, api_key, raw_dir, args.retries, args.workers)
    features = build_features(metadata, records, output_dir / "features.csv")
    audit = audit_forecast_availability(audit_frame)
    manifest = {
        "schema_version": 1,
        "competition_eligible": True,
        "provider": "Korea Meteorological Administration",
        "dataset": "KMA 5 km village short-range operational forecast",
        "source_type": "operational_forecast_archive",
        "documentation_url": DOCUMENTATION_URL,
        "license": "KMA public data/API terms; source attribution required",
        "license_url": "https://apihub.kma.go.kr/apiInfo.do",
        "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "coverage": {
            "issues": int(metadata["data_available_kst_dtm"].nunique()),
            "targets": int(metadata["forecast_kst_dtm"].nunique()),
            "objects": len(records),
            "grid": {"nx": args.nx, "ny": args.ny},
            "categories": list(CATEGORIES),
        },
        "availability_evidence": {
            "method": "provider_schedule_with_conservative_lag",
            "conservative_delay_minutes": args.publication_delay_minutes,
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
