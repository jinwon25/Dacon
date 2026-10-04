# 1185 목표 R/F 레벨 전이 후속 연구 — 2026-08-17

## 결론

`submit_v27.zip` / Public **1157.9736407889**를 유지한다. 직전 시즌의
Futures/Regular 활동 레벨을 현재 행의 리그와 연결하는 v48 잔차 contrast는
두 과거 선택 기원에서 동일 recipe 합의를 만들었지만 late-2024에서 반전됐다.
제출 ZIP은 만들지 않았다.

## 중복 피처 감사

champion 추론 코드를 다시 확인한 결과 다음 변화 정보는 이미 포함돼 있었다.

- 정확한 당해 시즌 pitcher/batter n·success count와 posterior
- 당해 시즌 성공률과 과거 prior의 logit 차이
- career/season 표본 신뢰도와 posterior 표준편차
- 최근 1/3/5경기 성공·middle 변화와 disagreement
- 시즌 월 대비 투구 pace, 고부하 여부
- 신규 투수와 한 시즌 이상 공백 후 복귀 여부

따라서 경력 단계, 복귀, workload를 이름만 바꿔 재실험하지 않았다. 반면
직전 시즌 주 활동 레벨과 현재 R/F의 전이는 기존 EDA에서만 다뤘고 champion
피처나 잔차 lookup에는 명시적으로 들어 있지 않았다.

## 전이군 진단

직전 시즌 투구 행의 80% 이상이 R이면 `R`, F이면 `F`, 그 외는 `MIX`,
기록이 없으면 `NEW`로 정의했다. 2023·2024 F에서 pitcher `R→F` 성공률은
각각 `0.4959`, `0.4845`로 F 전체보다 약 `+0.0245`, `+0.0253` 높았다.

v27 residual을 확인하면 2024 `R→F` F 행은 평균 `target - prediction =
+0.0091`이었고 8개 월 중 6개가 양수였다. 하지만 late-2023에는 F 전체의
절대 calibration이 크게 달랐으므로 전이군의 절대 성공률을 다음 시즌에
전달하는 방식은 사용하지 않았다.

## v48 — domain-centered level-transition contrast

각 source OOF에서 다음 값만 학습했다.

`EB mean(target - parent | transition, domain) - mean(target - parent | domain)`

즉, 리그 전체의 시간 drift를 제거하고 같은 domain 안에서 전이군이 parent보다
상대적으로 높거나 낮은 정도만 다음 시즌으로 전달했다. 그룹 shrinkage alpha는
`1000`으로 고정했다. pitcher, batter, `0.75/0.25` crossed 신호만 사용했다.

시간 축은 다음처럼 완전히 전진시켰다.

1. late-2021 wave0 OOF residual → late-2022 선택
2. late-2022 OOF residual → late-2023 선택
3. late-2023 v27 residual → full/late-2024 동결 평가

동일 signal·domain·weight가 첫 두 축에서 모두 양수이고 월·적용 domain gate를
통과해야 했다. 120개 recipe 중 10개가 consensus gate를 통과했고, 선택 recipe는
`batter_transition / R_ANCHOR / weight=0.20`이었다.

| 축 | gain vs parent | 양수 월 비율 | 최악 월 | 적용 domain gain |
|---|---:|---:|---:|---:|
| late-2022 선택 | +0.4963 | 100% | +0.1102 | 양수 |
| late-2023 선택 | +2.4437 | 100% | +0.7896 | 양수 |
| full-2024 | +0.1506 | 71.4% | -0.7755 | +0.8550 |
| late-2024 | **-0.5598** | **0%** | -0.6593 | **-3.2019** |

full-2024의 작은 양수는 3–7월에 집중됐고 8·9월은 모두 음수였다. 특히 원래
도메인 가설인 pitcher `R→F`/F가 아니라 batter/R_ANCHOR가 선택된 것은 작은
그룹 후보군 안에서도 recipe 선택이 흔들린다는 증거다. 2024 결과를 본 뒤
pitcher/F로 route를 바꾸거나 alpha·weight를 다시 고르지 않는다.

2024 전이군 target 진단이 실험 family 선택에 앞서 사용됐으므로 full-2024를
완전히 pristine한 family-level audit이라고 부르지 않는다. 다만 signal·route·
weight 자체는 late-2022/late-2023에서만 선택됐고, late-2024 반전은 명확한
기각 증거다.

## 누수·재현성

- 레벨 분류는 직전 시즌의 target-free `game_type` 빈도만 사용한다.
- contrast는 source OOF target과 source OOF parent로만 학습한다.
- query/evaluation label이나 다른 evaluation 행 집계를 사용하지 않는다.
- test에서는 2024 train으로 고정된 player transition lookup을 현재 행에만 매핑할
  수 있지만, 승격 실패로 패키징하지 않았다.
- OOF·모델·진단 NPZ는 `artifacts/`에만 저장하고 Git에 포함하지 않는다.

## 재현

```powershell
python -m src.archive.v48_level_transition_contrast --project . --output-dir artifacts/v48_level_transition_20260817_01
python -m pytest -q
```

재시도 금지 목록에는 level threshold, group alpha, pitcher/batter 혼합비,
domain route 및 weight의 추가 탐색을 포함한다.
