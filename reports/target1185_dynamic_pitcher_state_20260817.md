# 1185 목표 동적 투수 상태 감사 — 2026-08-17

## 결론

공식 `asof_pitcher_*` 누적값과 과거 시즌 투수 잠재 상태를 결합한 v51은 선택
단계에서 탈락했다. 16개 고정 후보 중 2022와 late-2023의 평균·월·도메인 gate를
동시에 통과한 후보는 0개였다. 2025 artifact와 제출 ZIP은 만들지 않으며 champion은
`submit_v27.zip` 그대로다.

| 구간 | gain vs 해당 v27 analogue | 양의 월 비율 | 최악 월 |
|---|---:|---:|---:|
| 2022 선택 | +0.0172 | 기준 미달 | -2.2601 |
| late-2023 선택 | +0.1003 | 기준 미달 | -0.0840 |
| full-2024 진단 | +0.0329 | 25.0% | -1.5611 |
| late-2024 진단 | -0.2273 | 0.0% | -0.4316 |

선택 gate를 통과한 후보가 없으므로 표의 최선 레시피
`ar_k30_w025 / R_CORE`는 배포 후보가 아니라 family 진단용 대표값이다.

## 방법과 누수 통제

1. 완료된 각 시즌의 투수 성공률을 해당 시즌 리그 평균으로 smoothing `200`만큼
   수축하고 league-centred logit 상태로 바꿨다.
2. 예측 연도보다 엄격히 이전인 연속 시즌의 투수 상태 쌍으로만 zero-intercept
   AR(1)을 적합하고 `rho`를 `[0, 1]`로 제한했다.
3. 각 행의 공식 career-to-date `n/rate`에서 직전 시즌 종료 career state를 빼
   현재 시즌의 직전 투구까지 표본을 복원했다.
4. AR prior와 전년도 상태 prior가 현재 시즌 표본 증가에 따라 자동 감쇠하도록
   strength `30/100`, 고정 가중치 `0.25/0.50` 네 후보만 사용했다.
5. 동일한 state/strength/weight/domain 레시피를 2022와 late-2023에서 고정한 뒤
   full/late-2024를 진단했다. 평가 행 간 집계나 외부 성공률은 사용하지 않았다.

추정된 AR 지속성은 2022 `0.3771`, 2023 `0.4497`, 2024 `0.3632`로 양수였지만,
행 단위 보정의 평균 절댓값은 2024에서 `ar_k30=0.000934`에 불과했다. 현재 v27의
exact current-season 상태 피처가 이미 대부분의 투수 정보를 흡수해 추가 보정의
독립 상한이 매우 작다는 해석과 일치한다.

## 결정과 재시도 금지

- AR smoothing/ridge/strength/weight 미세조정은 하지 않는다.
- late-2024를 본 뒤 부호를 뒤집거나 월·도메인을 좁히지 않는다.
- batter AR 또는 pitcher+batter 결합도 공개 원 실험과 현 저장소 v31에서 모두
  타자 상태 추가 이득이 없었으므로 반복하지 않는다.
- train-only global drift는 이미 `calibration.py`, `rolling_drift.py`,
  `reports/base_rate_forecasts.csv`에서 충분히 검증됐다. 2024 최신 fold가 악화되거나
  효과가 `5e-6` 수준이어서 외부 2025 평균을 이용한 shift로 재개하지 않는다.

독립 패턴 참고:
<https://github.com/mk-isos/lg-aimers-9-pitch-control/blob/main/experiments/train_exp072_dynamic_pitcher_state.py>

## 재현

```powershell
python -m src.v51_dynamic_pitcher_state
python -m pytest tests/test_v51_dynamic_pitcher_state.py -q
```

수치 산출물은 `artifacts/v51_dynamic_pitcher_state_20260817_01/`에만 저장되며
Git에 포함하지 않는다.
