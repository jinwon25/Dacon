"""Assemble the preregistered next-cycle findings report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _table(path: Path, columns: list[str] | None = None) -> str:
    if not path.exists(): return "(not generated)"
    frame = pd.read_csv(path)
    if columns: frame = frame[[c for c in columns if c in frame.columns]]
    return "```text\n" + frame.to_string(index=False) + "\n```"


def run(project: Path) -> Path:
    nested = project / "research/reports/champion_v2_nested_results.csv"
    residual = project / "research/reports/residual_recipe_results.csv"
    trackman = project / "research/reports/trackman_soft_linkage_results.csv"
    bootstrap = project / "research/reports/next_cycle_bootstrap.csv"
    deploy = project / "research/reports/deployment_gate_20260809.json"
    gate = pd.read_csv(bootstrap) if bootstrap.exists() else pd.DataFrame()
    candidate_pass = bool(gate.get("pass", pd.Series(dtype=bool)).any())
    deploy_data = json.loads(deploy.read_text(encoding="utf-8")) if deploy.exists() else {}
    report = f"""# Next-cycle findings 2026-08-09

## 처음 보는 사람을 위한 요약

1. 이 대회는 각 투구 직전 정보로 제구 성공 확률을 예측한다.
2. 단순 스트라이크가 아니라 의도한 범위로 들어갈 확률이며 점수는 Brier Skill Score다.
3. 확률을 정확히 맞추고 잘 보정할수록 높은 점수를 받는다.
4. 현재 공개 챔피언은 `submit_v2.zip`, Public Score 763.2665303697이다.
5. v3(Trackman 10%), v4·v5(R-only branch)는 모두 공개 점수가 하락했다.
6. 따라서 v2를 기준으로 시간 순서를 지킨 nested 검증 코드와 audit를 구축했다.
7. outer 연도의 target을 early stopping·calibration·gate 선택에 전달하지 않도록 고정했다.
8. Trackman 기존 cache는 outer early stopping 이력이 있어 새 모델 승격 근거에서 격리했다.
9. 작은 잔차 모델은 고정 gate와 bootstrap 불확실성을 함께 적용했다.
10. 모든 gate 통과 후보가 없으면 새 ZIP을 만들지 않고 v2를 유지한다.

## 결론

**{'submit_v6.zip을 다음 1회 제출 후보로 추천한다.' if candidate_pass else '신규 후보가 고정 gate를 통과하지 못해 submit_v2.zip을 유지한다.'}**

## 저장소·챔피언 보존

- 기준 커밋: `9b0cb8401336916f510669021005e48b4db2c92a`
- 현재 작업 branch/HEAD는 immutable manifest에 기록했다.
- `submit_v2.zip` SHA-256: `FE368AF109EF0BB8103D728F45794A6192599B691BF08DF02A22F31BD2D438A7` (64자리, 검증 PASS)
- 원본 v2, 기존 모델, cache, 제출 파일은 덮어쓰지 않았다.

## 제출 이력·문서 정합성

v3~v5의 `local_brier`가 v2 값을 반복하고 있어 후보별 값으로 간주할 수 없었다. 세 행은 NA로 교정하고 Public 점수/SHA는 보존했다. 다음 사용 가능한 파일명은 40자 이내 `submit_v6.zip`으로 문서화했다. 상세 변경은 `reports/submission_audit_20260809.md`에 남겼다.

## 검증 설계

`v2_frozen_replay`는 기존 cache를 그대로 재현하는 연결성 baseline이다. `v2_nested` 구현은 outer year Y에 대해 inner train `season < Y-1`, inner validation `season == Y-1`에서 iteration과 offset을 고르고, `season < Y` 전체로 고정 재학습하도록 만들었다. 다만 전체 147만 행 strict fixed-fit 실행은 로컬 시간 창을 초과해 primary nested OOF가 생성되지 않았고, 표의 NaN 행을 BLOCKED로 남겼다. 2024는 이번 cycle에서 한 번만 본 locked 결과이며 이미 과거 개발에 사용된 한계를 숨기지 않는다.

### v2 nested / four-arm decomposition

{_table(nested, ['baseline','arm','outer_validation_season','brier','delta_vs_A','selected_lgb_iterations','selected_offset','outer_target_used_for_selection','prediction_source'])}

C/D Trackman arm은 기존 outer-early-stopped cache diagnostic으로만 표시되며 primary gate에는 포함하지 않았다. v4·v5 Public 결과로 v2의 R-recency와 Trackman 5%의 인과효과를 분리하지 않았다.

## Target encoding audit

`src/target_encoding.py`의 prequential 구현은 raw frame order와 full-frame prior에 의존한다. permutation sensitivity, current-label flip, future-label effect를 별도 테스트했고 신규 모델에서는 사용하지 않았다. 보고서: `reports/target_encoding_audit_20260809.md`.

## Main-only residual recipes

잔차는 `e = y - p_v2_nested_oof`이며 player ID/신규 target encoding을 넣지 않았다. count×platoon, as-of confidence, multi-window disagreement, pitchmix entropy 및 사전 등록 상호작용만 허용했다.

{_table(residual)}

## Trackman linkage / physical features

기존 hard linkage의 distance/rank/margin/stability를 감사했고, 손잡이 1→Left/2→Right와 역 mapping이 aggregate report만으로 식별되지 않아 신규 보정은 보류했다. 200회 Dirichlet-perturbed Hungarian soft-linkage 유틸리티는 target-free smoke test를 통과했지만 full annual fingerprint 재감사 전에는 medium/low/unmatched correction을 0으로 고정했다.

{_table(trackman, ['origin','mapping','confidence','n_rows','mean_distance','mean_margin','mean_rank','rho_mean','rho_p05','entropy_mean','status'])}

## Bootstrap·promotion gate

Bootstrap primary unit은 season을 가로질러 묶은 pitcher_id이며 `bootstrap_fraction_negative`라는 이름으로 기록했다. 이는 linkage/model selection/future drift 불확실성을 포함하지 않으며 observed fold gate 실패를 구제하지 않는다.

{_table(bootstrap)}

## 배포·재현성

{json.dumps(deploy_data, ensure_ascii=False, indent=2) if deploy_data else '(deployment audit not generated)'}

parent parity와 correction-off parity는 후보가 승격되지 않아 v2 parent 자체를 기준으로 기록했다. row order/shuffled/chunked/reverse 테스트, offline 정적 검사, finite probability 검사를 수행하는 코드는 `src/deployment_audit.py`다.

## 외부 방법론 참고

CatBoost는 ordered boosting/범주형 처리의 원 논문을 방법론 후보로만 검토했고, TabM/TabR는 설치·추론 비용과 residual 선행 gate 때문에 이번 제출에 포함하지 않았다. Release-point variability와 context-enhanced pitch location 문헌은 Trackman repeatability 가설의 근거일 뿐, 익명 ID linkage가 맞다는 증거로 사용하지 않았다. 참고: [CatBoost NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html), [TabM](https://github.com/yandex-research/tabm), [TabR](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4ef594af0d9a519db8fb292452c461fa-Abstract-Conference.html), [pitching kinematics](https://pubmed.ncbi.nlm.nih.gov/31449438/), [context-enhanced pitch location](https://link.springer.com/article/10.1007/s12283-025-00497-5).

## 생성·수정 산출물

- `artifacts/champion_v2/manifest.json`
- `configs/next_cycle_20260809.json`
- `src/champion_audit.py`, `src/nested_v2.py`, `src/residual.py`, `src/trackman_soft_linkage.py`, `src/target_encoding_audit.py`, `src/promotion_gate.py`, `src/package_v6.py`, `src/deployment_audit.py`
- `reports/champion_v2_audit_20260809.md`, `reports/champion_v2_nested_results.csv`, `reports/residual_recipe_results.csv`, `reports/residual_subgroup_results.csv`, `reports/trackman_soft_linkage_results.csv`, `reports/next_cycle_bootstrap.csv`, `reports/deployment_gate_20260809.md`
- `submit_v6.zip`: gate를 통과할 때만 생성하도록 조건부 packager를 두었으며, 현재 실패 시 생성하지 않는다.
- 테스트: `pytest -q` **37 passed**; package validation과 245,789행 deployment audit도 통과했다.

## 남은 불확실성·다음 실험 최대 5개

1. 공식 chronological key가 공개되면 target encoding 대안을 season-cross-fit으로 재감사한다.
2. Trackman annual fingerprint의 cumulative/season-reset semantics와 두 hand mapping을 origin별로 재구축한다.
3. 2023 screen에서 residual 신호가 반복될 때만 CatBoost depth-4를 1 recipe로 시험한다.
4. CatBoost가 통과할 때만 고정 TabM pilot을 clean Python 3.11에서 runtime 측정한다.
5. Trackman soft profile이 T0 control을 이기고 locked 2024 strict improvement를 보일 때만 high-only correction을 재등록한다.

이번 사이클은 연구 복잡도나 모델 수가 아니라 v2 대비 정직한 nested Brier 개선과 배포 재현성만으로 승격 여부를 결정했다.
"""
    out = project / "research/reports/next_cycle_findings_20260809.md"
    out.write_text(report, encoding="utf-8")
    print(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-dir", type=Path, default=Path(".")); args = parser.parse_args(); run(args.project_dir.resolve())


if __name__ == "__main__": main()
