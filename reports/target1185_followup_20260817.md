# 1185 목표 후속 연구 결과 — 2026-08-17

> **2026-08-18 정정**: 이 보고서는 `submit_v27.zip`을 champion(frozen baseline)으로 가정하지만, 공식 DACON 제출 이력 재대조 결과 실제 champion은 `submit_v26.zip`이다(Public `1157.9736407889`, 제출 ID `51773`; v27은 `1156.6153781694`로 champion보다 낮다). 이 보고서의 v27 대비 gain 비교는 champion(v26) 기준으로 다시 확인하기 전까지 그대로 인용하지 않는다. 근거: [`target1170_followup_20260817.md`](target1170_followup_20260817.md) 상단, [`../notebooks/v26_champion_reproduction.ipynb`](../notebooks/v26_champion_reproduction.ipynb).

## 결론

이번 사이클에서 `submit_v27.zip`을 넘는 신규 제출 후보는 만들지 않았다.
현재 Public champion은 **1157.9736407889**이며 확인 당시 10위다. 1185까지
`+27.0263592111`, 1200까지 `+42.0263592111`이 남아 있다.

신규 다섯 실험은 모두 평가 행별 독립 추론을 유지했고, late-2023 선택과
full/late-2024 감사를 분리했다. 선택축에서 큰 이득을 낸 lookup·ASOF 계열도
다음 연도에서 방향이 뒤집혔다. 낮은 분산의 확률 보정 역시 late-2024에서
실패했다. 따라서 Public 점수 확인을 위한 ZIP을 만들거나 제출하지 않았다.

## 고정 기준선과 재현 상태

| 항목 | 값 |
|---|---:|
| champion | `submit_v27.zip` |
| Public | 1157.9736407889 |
| SHA-256 | `CEFC91F4F8EBBA32EE8025816319F32C95023CB59EA70741176A98FB7A172385` |
| 245,789행 추론 | 56.882초 |
| peak RSS | 약 1.4GB |
| 배치 불변성 최대 오차 | `1.11e-16` |
| 이번 사이클 전체 테스트 | **137 passed** |

v27은 v22의 저분산 domain/ASOF prior 위에 v25 R_ANCHOR post-break 직접모형
방향을 10% 적용한 모델이다. v25의 세 사전 시간축 gain은 `+31.4866`,
`+1.7500`, `+0.8211`이고, Public에서는 v22 대비 `+2.7857` 전이됐다.

## 신규 실험 결과

| 버전 | 가설 | 선택축 | full-2024 감사 | late-2024 감사 | 판정 |
|---|---|---:|---:|---:|---|
| v30 | 기존 106개 legal OOF 신호의 공분산·다양성 재선별 | 40개 gate 통과 | 양의 독립감사 후보 0개 | 양의 양축 후보 0개 | 기각 |
| v31 | 선수→도메인→손→압박/카운트 동적 계층 EB | +172.2589 | -15.8875 | -4.9781 | 기각 |
| v32 | 두 시간 절반에서 부호가 일치한 EB만 적용 | +21.8366 | -2.1482 | +1.8896 | 기각 |
| v33 | 표본수 축소 pitcher/batter ASOF의 log-odds 결합 | +148.8588 | -29.6894 | -16.6986 | 기각 |
| v34 | v27 identity 쪽으로 수축한 최신 확률 보정 | 선택 gate 0개 | early-2024 최선 +0.7896 | -2.0406 | 기각 |

### v30 — 기존 OOF bank 다양성

- 106개 season-forward 신호와 6,432개 저가중치 조합을 v27 위에서 평가했다.
- late-2023에서 사전 gate를 통과한 조합은 40개였지만, full-2024와
  late-2024에서 동시에 양수인 후보는 0개였다.
- 기존 PFD, failure-mode, exact-ASOF, residual bank의 단순 재블렌딩은 v27의
  잔차를 안정적으로 줄이지 못한다.

### v31 — 동적 계층 잔차

- 투수·타자 효과를 player → domain → 상대 손 → pressure/count 순으로 부분
  pooling하고, 월 반감기와 crossed 75/25 효과를 함께 검사했다.
- late-2023 최선은 batter-pressure adaptive, 1개월 반감기, eta 0.2였지만
  full-2024에서 월 양수 비율 12.5%, 최악 월 -48.16으로 반전했다.
- 최근 한두 달의 player/context 잔차를 2025로 외삽하는 계열은 재시도하지
  않는다.

### v32 — 시간 합의 EB

- source를 두 연속 시간 절반으로 나누고 두 효과의 부호가 같은 그룹만
  보정했다. v31보다 변동성은 줄었지만 full-2024 gain은 -2.1482였다.
- 단순 sign-consensus 또는 최소 절대효과 규칙만으로 연도 전이를 확보할 수
  없었다.

### v33 — 계층 log-odds ASOF prior

- Bayesian matchup/log5의 부분 pooling 아이디어를 공식 row-local ASOF rate에
  맞게 변형했다. 표본수가 적으면 source-domain 성공률로 축소하고, pitcher와
  batter posterior를 log-odds에서 결합했다.
- 선택 최선의 late-2023 F gain이 +1365.60으로 지나치게 컸고, full-2024의
  모든 도메인이 음수였다. 2023 후반 regime에 맞은 선택 편향으로 판정한다.
- ASOF prior의 선형/비선형 재조합과 강도 미세조정은 중단한다.

### v34 — 최신 확률 보정

- late-2023에 `target - v27` ridge 보정식을 맞추고 모든 계수를 identity인 0
  쪽으로 수축했다. early-2024만으로 family/강도/route를 선택하고 late-2024는
  보류했다.
- 300개 후보 중 early-2024 사전 gate 통과 후보가 없었다. 가장 보수적인
  R_ANCHOR 상수 이동도 late-2024 해당 도메인에서 -11.6710이었다.
- v27 이후 추가 Platt/beta/bin/domain calibration은 재시도하지 않는다.

## 누수·과적합 감사

- 각 correction table과 calibration map은 라벨이 있는 source 구간에서만
  학습하고 이후 구간에 고정 적용했다.
- 평가 행의 target, 평가 행 집계, 다른 평가 행의 빈도·순서·그룹 정보는
  사용하지 않았다.
- v30~v33은 late-2023에서만 선택하고 2024 감사를 후보 선택에 사용하지
  않았다. v34는 late-2023 fit → early-2024 selection → late-2024 audit의
  세 단계로 분리했다.
- 신규 Public 제출은 하지 않았고 v27/v25 가중치를 Public에 맞춰 조정하지
  않았다.

## 재시도 금지 목록

다음은 새 독립 정보나 더 오래된 완전 OOF가 생기기 전에는 다시 실행하지
않는다.

1. v27 위 기존 106개 OOF bank의 단일·저가중치 재블렌딩
2. 최근 월 반감기를 둔 player/context residual lookup
3. 두 기간 부호 합의만 사용한 EB lookup
4. pitcher/batter ASOF rate의 log-odds·shrinkage 재조합
5. v27 위 bin/Platt/beta/domain/affine 확률 보정
6. v25 R_ANCHOR의 손잡이·카운트 route 또는 10% 주변 가중치 미세조정

## 다음 우선순위

현재 저장소 안의 동종 신호는 포화됐다. 다음 유효 작업은 아래 둘 중 하나다.

1. **완전한 2022·2023·2024 champion analogue OOF**: v19→v20→v21→v22→v27
   계보를 각 연도 이전 데이터만으로 재생성해, 신호 선택과 최종 감사를 서로
   다른 연도로 분리한다. 현재 2022에는 일부 state/mode OOF만 있어 전체
   champion 잔차를 비교할 수 없다.
2. **독립 팀 모델 row-level 예측**: 동일 행 순서와 타깃을 검증한 뒤 v27과의
   오차 공분산을 측정한다. v30 결과상 현재 저장소의 기존 bank를 다시 섞는
   것보다 생성 구조가 다른 예측이 필요하다.

둘 중 어느 것도 준비되지 않은 상태에서는 제출 횟수를 쓰지 않는다. 제출
우선순위는 계속 `submit_v27.zip` 단독이며, 백업은 `submit_v25.zip`이다.

## 방법론 참고

- Han et al. (2024), *Model Assessment and Selection under Temporal
  Distribution Shift*: https://proceedings.mlr.press/v235/han24b.html
- Doo & Kim (2018), hierarchical Bayesian matchup/log5:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6192592/
- Brill et al. (2025), xCTRL intent/execution separation:
  https://arxiv.org/abs/2508.19184

xCTRL은 현재 투구 위치와 의도 위치가 필요한 접근이다. 이 대회 입력에는 그
정보가 없으므로 player/context 잠재구조의 동기로만 참고했고 직접 구현하지
않았다.

## 재현 명령

```powershell
python -m src.v30_diverse_covariance_screen --project .
python -m src.v31_dynamic_hierarchical_residual --project .
python -m src.v32_temporal_consensus_eb --project .
python -m src.v33_logodds_asof_prior --project .
python -m src.v34_temporal_v27_calibration --project .
python -m pytest -q
```

모델·OOF·제출 ZIP과 원데이터는 Git 대상이 아니다. 위 명령의 생성물은
`.gitignore` 아래 `artifacts/`에만 남긴다.
