"""Write the auditable Top-1100 screen tables and final decision report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.archive.data import read_main
from src.metrics import brier_score, brier_skill_score


def decompose(y: np.ndarray, p: np.ndarray, bins: int = 20) -> dict:
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    bucket = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    r = float(y.mean()); rel = res = 0.0
    for b in np.unique(bucket):
        mask = bucket == b; w = float(mask.mean()); yp = float(y[mask].mean()); pp = float(p[mask].mean())
        rel += w * (pp - yp) ** 2; res += w * (yp - r) ** 2
    return {"brier": brier_score(y, p), "uncertainty": r * (1 - r), "reliability": rel, "resolution": res}


def run(project: Path) -> None:
    out_dir = project / "research/reports/top1100"; art_dir = project / "artifacts/top1100"
    train = read_main(project / "data/train.csv")
    valid = train.loc[train["season"] == 2024].copy().reset_index(drop=True)
    y = valid["control_success"].to_numpy(dtype=float)
    import script as champion_script
    v2_path = art_dir / "v2_package_exact_2024.npy"
    if v2_path.exists():
        v2 = np.load(v2_path)
    else:
        v2 = np.asarray(champion_script.predict_dataframe(valid.drop(columns=["control_success"])), dtype=float)
        np.save(v2_path, v2)
    preds = {"v2_package_exact_legacy_2024": v2}
    for path in sorted(art_dir.glob("*_screen_2024.npy")):
        preds[path.stem.replace("_2024", "")] = np.load(path)
    rows = []
    dec = []
    for name, p in preds.items():
        rows.append({"model": name, "outer_validation_season": 2024, "n_rows": len(y), "brier": brier_score(y, p), "score_equivalent": brier_skill_score(y, p), "delta_vs_v2_package_exact": brier_score(y, p) - brier_score(y, v2)})
        d = decompose(y, p); d["model"] = name; dec.append(d)
    pd.DataFrame(rows).to_csv(out_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(dec).to_csv(out_dir / "brier_decomposition.csv", index=False)

    subgroup_rows = []
    valid["count_state"] = valid["balls_before"].astype(str) + "-" + valid["strikes_before"].astype(str)
    valid["pitcher_history_bucket"] = pd.cut(valid["asof_pitcher_n"].fillna(0), [-1, 0, 29, 199, np.inf], labels=["0", "1-29", "30-199", "200+"])
    valid["pitcher_known"] = np.where(valid["asof_pitcher_n"].fillna(0) > 0, "known", "unknown")
    for group in ["game_type", "count_state", "pitcher_history_bucket", "pitcher_known"]:
        for value, idx in valid.groupby(group, observed=True).groups.items():
            yi = y[np.asarray(idx)]
            for name, p in preds.items():
                pi = p[np.asarray(idx)]
                subgroup_rows.append({"group": group, "value": str(value), "model": name, "n_rows": len(idx), "brier": brier_score(yi, pi), "delta_vs_v2_package_exact": brier_score(yi, pi) - brier_score(yi, v2[np.asarray(idx)])})
    pd.DataFrame(subgroup_rows).to_csv(out_dir / "subgroup_metrics.csv", index=False)
    names = list(preds)
    corr = pd.DataFrame(np.corrcoef(np.vstack([preds[n] for n in names])), index=names, columns=names)
    corr.to_csv(out_dir / "model_diversity.csv")

    frozen = pd.read_csv(project / "research/reports/champion_v2_nested_results.csv")
    drow = frozen[(frozen["baseline"] == "v2_frozen_replay") & (frozen["arm"] == "D_full_v2")]
    frozen_2024 = float(drow.loc[drow["outer_validation_season"] == 2024, "brier"].iloc[0]) if not drow.empty else float("nan")
    contract = json.loads((art_dir / "data_contract.json").read_text(encoding="utf-8"))
    lgb = next((r for r in rows if r["model"] == "p0a_lgb_state_screen"), None)
    cb = next((r for r in rows if r["model"] == "categorical_catboost_screen"), None)
    fm = next((r for r in rows if r["model"] == "field_aware_fm_screen"), None)
    report = f"""# Top-1100 structural modeling — final decision (2026-08-09)

## 1. 대회와 현재 상황 요약

1. 각 투구 직전의 경기 상황·선수 이력으로 `control_success=1` 확률을 예측한다.
2. 평가는 Brier Skill Score이므로 적중률보다 확률의 보정과 조건부 분해능이 중요하다.
3. 현재 보존된 champion은 `submit_v2.zip`, Public 763.2665303697이다.
4. 763에서 1100으로 가려면 base rate 약 0.5 기준 Brier를 약 0.00084 줄여야 한다.
5. 하나의 전역 calibration이나 blend weight만으로 닫기 어려운 격차다.
6. 이번 cycle은 2019–2023 train, 2024 locked confirmation으로 screen했다.
7. 2024는 과거에 반복 노출된 개발 구간이므로 virgin holdout으로 부르지 않는다.
8. test 행 간 통계·순서·검색·적응은 사용하지 않는다.
9. 원본 외부 야구 데이터는 사용하지 않고, 문헌은 방법론 근거로만 사용한다.
10. 고정 gate를 통과한 후보만 `submit_v6.zip`으로 만들 수 있다.

## 2. 데이터·champion·git 감사

- train: {contract['files']['train']['rows']}행 × {contract['files']['train']['columns']}열, SHA-256 `{contract['files']['train']['sha256']}`
- formal test sample: {contract['files']['test']['rows']}행 × {contract['files']['test']['columns']}열 (전체 hidden test 재구성 아님)
- Trackman: {contract['files']['trackman_history']['rows']}행 × {contract['files']['trackman_history']['columns']}열, SHA-256 `{contract['files']['trackman_history']['sha256']}`
- `submit_v2.zip`: 8,880,825 bytes, SHA-256 `FE368AF109EF0BB8103D728F45794A6192599B691BF08DF02A22F31BD2D438A7` — immutable 보존·검증 PASS.
- 현재 저장소 HEAD는 BARAM 작업 branch이며 기준 commit `9b0cb8401336916f510669021005e48b4db2c92a`는 다른 aimers branch에 존재하지만 현재 HEAD의 조상은 아니다. 사용자 변경분 때문에 branch 전환/reset은 하지 않았다.

## 3. 핵심 구조 발견

1. `asof_pitcher_n` 전이는 1,473,508건 전수에서 100% 일치했다. 현재 season 시작 전 snapshot을 사용해 season 노출·성공 count를 복원할 근거가 확보됐다. 표시 rate의 round-trip 오차 p99.9는 약 0.0058 count 단위, max 약 0.0077이었다.
2. target-free 손잡이 집계는 `1=Left, 2=Right`가 반대 mapping보다 모든 season aggregate share error가 작았다. 이는 aggregate 의미 확인이지 player identity 정답률은 아니다.
3. main game block ↔ Trackman candidate alignment는 4,810/7,228 main block을 매칭했지만 실제 high rule 6.82%가 placebo 7.26–8.27%보다 좋지 않았고 median distance도 개선되지 않았다. 따라서 Trackman 물리 profile 승격을 중단했다.

## 4. baseline·validation 부채

`v2_frozen_replay`는 과거 package recipe parity용 진단 baseline이다. `v2_nested`는 outer target을 early stopping·offset·weight에 쓰지 않는 builder를 만들었지만 전체 strict OOF가 local runtime을 초과해 완성되지 않았다. 따라서 신규 screen을 champion 후보나 1100 근거로 부르지 않는다. frozen D full-v2의 2024 Brier는 {frozen_2024:.9f}이며 legacy diagnostic이다.

## 5. 구조 모델 screen (2024, target 미사용)

| model | Brier | score-equivalent | 판단 |
|---|---:|---:|---|
| v2 package exact (legacy, target-overlap diagnostic) | {rows[0]['brier']:.9f} | {rows[0]['score_equivalent']:.1f} | 비교용, promotion 금지 |
| P0-A full-state LightGBM (120만 train rows) | {lgb['brier'] if lgb else float('nan'):.9f} | {lgb['score_equivalent'] if lgb else float('nan'):.1f} | v2 legacy 대비 개선 아님 |
| categorical-safe CatBoost (18만 screen rows) | {cb['brier'] if cb else float('nan'):.9f} | {cb['score_equivalent'] if cb else float('nan'):.1f} | family screen 탈락 |
| field-aware FM pilot (25만 rows, 5 epochs) | {fm['brier'] if fm else float('nan'):.9f} | {fm['score_equivalent'] if fm else float('nan'):.1f} | 수렴·보정 실패, 종료 |
| one-hot SGD dynamic-logit screen (30만 rows) | 0.343705606 | 0.0 | 종료 |

The package prediction is trained with all seasons and is not an honest outer baseline; the table is therefore a diagnostic comparison only. No candidate satisfies the champion gate (2024 ΔBrier ≤ −0.00010, four-fold weighted ΔBrier ≤ −0.00010, worst fold ≤ +0.00005, cluster bootstrap ≥ .95).

## 6. gate·패키지 판정

- outer nested OOF: **FAIL/BLOCKED** — full strict run not available.
- P0-B linkage placebo separation: **FAIL** — no Trackman promotion.
- structural model screen: **FAIL** — no repeatable honest improvement.
- subgroup/correction/bootstrap gate: **NOT RUN / NOT ELIGIBLE**.
- deployment parity for existing v2: **PASS** (previous package audit; 245,789 rows within seconds, row-order checks passed).
- candidate package: **NOT CREATED**. `submit_v6.zip` does not exist and no DACON submission was made.

## 7. 최종 결론

**신규 후보가 고정 gate를 통과하지 못해 submit_v2.zip을 유지한다.**

이번 cycle의 유효한 산출물은 데이터 계약, as-of state 복원 감사, target-free alignment audit, 구조 모델 screen과 재현 테스트다. CatBoost/FＭ/LightGBM screen 결과만으로 Public score 개선을 주장하거나 submit_v6을 만드는 것은 통계적으로 정직하지 않다.

## 8. 다음 실험 최대 5개

1. 기준 branch에서 4-fold strict nested v2를 run cache와 고정 iteration으로 완주하고 row-level OOF hash를 생성한다.
2. P0-A state를 2019–2023 full train의 ordered LightGBM/DeepFM에 넣되 2021–2023 inner screen → 2024 one-time lock 순서를 지킨다.
3. main에 공식 game_date/game_id가 제공되는지 운영진 Q&A로 확인한 뒤에만 pitch-level Trackman alignment를 재개한다.
4. FM pilot의 saturation 원인을 고친 단일 low-learning-rate seed만 사전 등록해 재시험하고, 개선 없으면 family를 종료한다.
5. strict OOF가 확보되기 전에는 residual stack·calibration·submit_v6을 만들지 않는다.
"""
    (out_dir / "final_decision.md").write_text(report, encoding="utf-8")
    (out_dir / "top1100_findings_20260809.md").write_text(report, encoding="utf-8")
    print("wrote final_decision.md, top1100_findings_20260809.md and audit tables")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--project-dir", type=Path, default=Path(".")); run(parser.parse_args().project_dir.resolve())
