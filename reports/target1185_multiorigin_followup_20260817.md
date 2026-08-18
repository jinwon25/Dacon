# 1185 목표 다중 원점·최신 시즌 후속 연구 — 2026-08-17

## 결론

이번 사이클에서도 `submit_v27.zip`을 넘는 신규 제출 후보는 만들지 않았다.
현재 Public champion은 **1157.9736407889**이며, 1185까지 `+27.0263592111`,
1200까지 `+42.0263592111`이 남아 있다.

기존 OOF의 시간 계보를 다시 감사한 뒤 다중 원점 합의, 최신 시즌 직접
CatBoost 잔차, 현재 투구 TrackMan을 학습 시 teacher에만 쓰는 privileged
distillation을 순서대로 검증했다. v38 TrackMan 학생만 독립 late-2024에서
평균 `+0.6842`였지만 9월과 모든 부트스트랩 하단이 음수였고 목표 최소
개선폭에도 미달했다. 따라서 감사 결과를 본 뒤 가중치를 바꾸거나 Public
제출로 재검증하지 않았다.

## OOF 계보 감사

- 2021 audit OOF는 저장소에 없다.
- 2022 audit 산출물은 state/mode, recent exact, exact screen, anchor 등 일부
  계열 6개뿐이다.
- champion analogue v20/v21은 직전 2021 동일 계보 OOF를 필요로 하므로,
  2022에서 완전한 v27 부모를 정직하게 재생성할 수 없다.
- 서로 다른 부모의 2022 예측을 v27 analogue라고 간주하지 않았다. v35에서는
  이를 `identity prefilter`로만 사용했고, v36에서는 동일 보정 recipe의 방향
  합의만 확인했다.

## 신규 실험 결과

| 버전 | 설계 | 선택 결과 | 보류 감사 결과 | 판정 |
|---|---|---:|---:|---|
| v35 | 2022 family prefilter → late-2023 pair 선택 → 2024 감사 | `+160.5170` | full `-18.1131`, late `-17.4751` | 기각 |
| v36 | 2022와 late-2023에서 동일 recipe·부호 합의 → 2024 감사 | `+55.2813`, `+9.2812` | full `-7.4490`, late `-4.8838` | 기각 |
| v37 | 2024 3~5월 CatBoost 잔차 → 6~7월 선택 → 8~10월 감사 | `+0.9296` | `-0.4424`, 3 seeds 모두 음수 | 기각 |
| v38 | 같은 월 분할의 TrackMan teacher / 안전 student 증류 | `+1.0327` | `+0.6842`, 최악 월 `-0.0292` | 보존·미승격 |

### v35 — family-balanced three-stage bank

48개 신호를 exact, mode, recent, state 네 family로 나눠 2022에서는 family별
상위 3개만 남겼다. late-2023에서 960개 단일 후보와 13개 저상관 pair를
평가했으며 178개가 선택 gate를 통과했다. 최종 F mode + R_CORE state pair는
선택축에서 컸지만, full-2024의 F gain이 `-81.7032`, late-2024의 F gain이
`-100.5478`로 반전했다. 서로 다른 부모의 2022 prefilter만으로 선택 편향을
막지 못했다.

### v36 — exact-recipe two-origin consensus

신호, 기준 방향, 도메인, 가중치가 완전히 같은 480개 recipe를 2022와
late-2023 양쪽에 적용했다. 46개가 두 원점의 모든 월 양수 gate를 통과했고,
`state::global_h4_l15_b075 / toward_parent / R_CORE / 0.2`가 선택됐다. 그러나
full-2024 R_CORE는 `-10.5745`, late-2024는 `-6.9752`였다. 부분 OOF의 두 원점
합의만으로 2024 regime 이동을 견디지 못했다.

### v37 — latest-season CatBoost residual

2024년 3~5월 `target-v27` 잔차를 안전한 행별 피처로 학습하고, 6~7월에
feature variant·route·eta를 선택했다. 선택된 player-ID/R_ANCHOR/0.1 recipe는
두 선택 월 모두 양수였지만 8~9월 감사 평균 `-0.4424`, 세 seed gain은
`-0.3632`, `-0.5261`, `-0.4870`이었다. 최신 시즌 직접 잔차 GBDT를 추가
튜닝하지 않는다.

### v38 — latest-season TrackMan privileged distillation

TrackMan 시퀀스 정렬 2024 행 216,555개 중 3~5월을 사용해 game-group 3-fold
safe/full teacher를 만들었다. 학생은 현재 투구의 구종·구속·회전·무브먼트를
보지 않고 공식 추론 안전 피처만으로 `full OOF - safe OOF`를 학습했다.
6~7월 선택은 ID 없는 학생을 R_ANCHOR에 0.4 적용하는 recipe였다.

- teacher privileged gain: 선택 학습 `+398.3587`, 재학습 `+354.9876`
- late-2024 평균 gain: `+0.6842`; R_ANCHOR active gain `+3.9130`
- 월: 8월 `+1.2776`, 9월 `-0.0292`
- seed gain: `+0.6867`, `+0.6987`, `+0.6666`
- bootstrap 5% 하단: pitcher `-0.4977`, batter `-0.2686`, crossed
  pitcher×batter `-1.2051`, block-2000 `-0.2996`
- crossed bootstrap 개선확률: `72.1%`

teacher가 현재 투구의 물리 정보로 성공 결과를 더 잘 설명하는 사실은
재확인됐지만, pre-pitch 학생으로 전이되는 신호는 작고 불확실하다. 감사
결과를 보고 0.4보다 작은 가중치나 다른 route를 고르는 것은 사후 선택이므로
실행하지 않는다.

## 누수·규칙·재현성 감사

- 모든 선택은 이후 감사 구간을 열기 전에 고정했다.
- 평가 행의 target, 다른 평가 행 집계·빈도·순서, test 전체 분포는 입력으로
  쓰지 않았다.
- v38의 현재 투구 TrackMan 값은 labelled source의 cross-fitted teacher에만
  사용했다. 선택·감사·최종 학생 입력의 TrackMan 열은 비어 있고 학생 feature
  목록에도 없다.
- 모델 seed 3개와 pitcher, batter, crossed, 시간 block 부트스트랩을 별도로
  확인했다.
- 실험 산출물과 모델은 `.gitignore` 아래 `artifacts/`에만 저장한다.

## 재시도 금지 및 다음 우선순위

새 독립 정보가 없으면 다음을 재시도하지 않는다.

1. 부분 2022 OOF를 이용한 family prefilter 또는 단순 다중 원점 recipe 합의
2. 2024 월 분할의 직접 CatBoost 잔차와 seed/eta 미세조정
3. v38 감사 결과를 이용한 TrackMan student의 route·가중치 사후 조정
4. 기존 2023→2024 표준 PFD, 과거 TrackMan 물리 profile, latent signal 재혼합

다음 고가치 입력은 동일 팀의 독립 row-level 예측 또는 2021부터 재생성한
완전한 champion analogue OOF다. 둘 중 하나가 생기기 전 제출 우선순위는
`submit_v27.zip` 단독, 백업은 `submit_v25.zip`으로 유지한다.

## 재현 명령

```powershell
python -m src.v35_three_stage_multibank --project .
python -m src.v36_two_origin_consensus --project .
python -m src.v37_latest_season_catboost_residual --project .
python -m src.v38_latest_trackman_pfd --project .
python -m pytest -q
```

주요 결과는 각각 `artifacts/v35_*`, `artifacts/v36_*`, `artifacts/v37_*`,
`artifacts/v38_*`의 `summary.json`에 있으며 Git에는 포함하지 않는다.
