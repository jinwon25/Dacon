from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.fetch_kma_um_context import (
    ContextRequestSpec,
    build_context_features,
    build_context_url,
    context_request_specs,
    parse_context_response,
    replace_targets_with_issue_offset,
    validate_context_response,
)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_kst_dtm": pd.date_range(
                "2024-01-01 01:00:00", periods=3, freq="h"
            ),
            "data_available_kst_dtm": pd.Timestamp("2023-12-31 13:00:00"),
        }
    )


def test_context_specs_cover_two_cycles_and_two_layers() -> None:
    specs, audit = context_request_specs(
        _metadata(), latitude=37.28, longitude=128.96
    )
    assert len(specs) == 4
    assert {(spec.cycle_rank, spec.data_kind) for spec in specs} == {
        (0, "U"),
        (0, "P"),
        (1, "U"),
        (1, "P"),
    }
    assert {spec.initialization_utc.hour for spec in specs if spec.cycle_rank == 0} == {
        12
    }
    assert {spec.initialization_utc.hour for spec in specs if spec.cycle_rank == 1} == {
        6
    }
    assert len(audit) == 2


def test_replace_targets_with_issue_offset_keeps_one_causal_target_per_issue() -> None:
    metadata = pd.concat(
        [
            _metadata(),
            _metadata().assign(
                data_available_kst_dtm=pd.Timestamp("2024-01-01 13:00:00")
            ),
        ],
        ignore_index=True,
    )
    result = replace_targets_with_issue_offset(metadata, -2.0)
    assert result["data_available_kst_dtm"].is_unique
    assert (
        result["forecast_kst_dtm"]
        == result["data_available_kst_dtm"] - pd.Timedelta(hours=2)
    ).all()
    assert replace_targets_with_issue_offset(metadata, None).equals(metadata)


def test_context_url_has_levels_and_redacts_key() -> None:
    spec = ContextRequestSpec(
        reference_kst=pd.Timestamp("2023-12-31 13:00", tz="Asia/Seoul"),
        initialization_utc=pd.Timestamp("2023-12-30 12:00", tz="UTC"),
        public_availability_utc=pd.Timestamp("2023-12-31 00:00", tz="UTC"),
        lead_start=27,
        lead_end=51,
        lead_step=3,
        latitude=37.28,
        longitude=128.96,
        cycle_rank=0,
        group="UMGL",
        nwp="N128",
        data_kind="P",
        levels_hpa=(850.0, 700.0),
    )
    url = build_context_url(spec, "private-key", redact=True)
    assert "private-key" not in url
    assert "%3Credacted%3E" in url
    assert "data=P" in url
    assert "level=850%2C700" in url


def test_context_parser_normalizes_pa_and_validates_profile() -> None:
    text = "\n".join(
        f"2023123012 2023123115 {variable} {level} {value}"
        for level, value in ((85000, 1.0), (70000, 2.0))
        for variable in (2002, 2003)
    )
    parsed = parse_context_response(text)
    assert set(parsed["level_hpa"]) == {850.0, 700.0}
    spec = ContextRequestSpec(
        reference_kst=pd.Timestamp("2023-12-31 13:00", tz="Asia/Seoul"),
        initialization_utc=pd.Timestamp("2023-12-30 12:00", tz="UTC"),
        public_availability_utc=pd.Timestamp("2023-12-31 00:00", tz="UTC"),
        lead_start=27,
        lead_end=27,
        lead_step=3,
        latitude=37.28,
        longitude=128.96,
        cycle_rank=0,
        group="UMGL",
        nwp="N128",
        data_kind="P",
        levels_hpa=(850.0, 700.0),
    )
    validate_context_response(parsed, spec)


def _write_record(
    tmp_path: Path,
    *,
    rank: int,
    kind: str,
    initialization: str,
    availability: str,
    offset: float,
) -> dict[str, object]:
    lines = []
    for valid, time_value in (("2023123115", 0.0), ("2023123118", 3.0)):
        levels = (10.0,) if kind == "U" else (850.0, 700.0)
        for level in levels:
            for variable, component in ((2002, 1.0), (2003, 2.0)):
                value = offset + time_value + component + level / 1_000.0
                lines.append(
                    f"{pd.Timestamp(initialization).strftime('%Y%m%d%H')} "
                    f"{valid} {variable} {level:g} {value}"
                )
    path = tmp_path / f"r{rank}_{kind}.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "path": path.as_posix(),
        "prediction_reference_kst": pd.Timestamp(
            "2023-12-31 13:00", tz="Asia/Seoul"
        ).isoformat(),
        "initialization_utc": pd.Timestamp(initialization).isoformat(),
        "conservative_public_availability_utc": pd.Timestamp(
            availability
        ).isoformat(),
        "cycle_rank": rank,
        "data_kind": kind,
    }


def test_build_context_features_interpolates_and_derives_run_change(tmp_path) -> None:
    records = [
        _write_record(
            tmp_path,
            rank=0,
            kind=kind,
            initialization="2023-12-30 12:00+00:00",
            availability="2023-12-31 00:00+00:00",
            offset=0.0,
        )
        for kind in ("U", "P")
    ] + [
        _write_record(
            tmp_path,
            rank=1,
            kind=kind,
            initialization="2023-12-30 06:00+00:00",
            availability="2023-12-30 18:00+00:00",
            offset=1.0,
        )
        for kind in ("U", "P")
    ]
    result = build_context_features(_metadata(), records, tmp_path / "features.csv")
    assert len(result) == 3
    assert result["kma_um_ctx_u10_r0"].tolist() == pytest.approx(
        [2.01, 3.01, 4.01]
    )
    assert result["kma_um_ctx_run_delta_u10"].tolist() == pytest.approx(
        [-1.0, -1.0, -1.0]
    )
    assert (result["kma_um_ctx_run_change10"] > 0).all()
    assert result["kma_um_ctx_shear10_850_r0"].notna().all()
    assert (tmp_path / "features.csv").is_file()
