# 그룹 3 KMA regime-conditioned power curve — 2026-07-26

## 결론

제출 36의 그룹 2 비핵심 상향 보완 실패를 반영해 그룹 2 확장을 종료하고,
공개 성공한 KMA UMRG 단조 power curve 위에서 그룹 3의 상층 물리 regime을
추가 검증했다. Q2에서 선택된 풍향 4-sector 후보는 희소성 계약을 적용한 H2에서
총점과 FiCR가 소폭 개선됐지만 시간 안정성·bootstrap·효과 크기 조건을 통과하지
못했다. 제출 CSV는 생성하지 않았다.

## 공개 결과 반영

- 제출: `1501731`
- 파일: `kma_group2_alpha2375_up_supplement_20260725.csv`
- score: `0.6439398345`
- 현재 최고 `1501487` 대비 score `-0.0001599771`
- 1-NMAE `-0.0000119700`, FiCR `-0.0003079842`

두 구성지표가 모두 하락했으므로 alpha 23.75% 핵심 밖의 상향 보완은
`rejected`로 닫았다. 현재 선택 제출은 계속
`kma_group2_overlay_alpha2375_20260725.csv`다.

## 실험 계약

`experiments/kma_group3_regime_power_curve.py`는 현재 KMA power-curve
incumbent의 그룹 3 lineage를 재현한 뒤 다음 여섯 물리 regime을 비교한다.

- 10 m 풍향 4-sector
- 10 m–850 hPa 시어 3구간
- 10 m–850 hPa 풍향 정렬 2구간
- 최신·직전 UM 사이클 변화 3구간
- 풍향 4-sector × 시어 2구간
- 풍향 4-sector × 정렬 2구간

각 regime isotonic curve는 전역 curve 쪽으로 표본 수 기반 수축을 적용한다.
Q1에서 curve를 학습하고 Q2에서 방향·coverage·출력비·혼합률을 선택한다.
선택 단계와 H2 모두 변경률 10%, 행별 추가 이동 1% 용량 상한을 적용한다.
H2는 H1 재학습 후 한 번만 평가하며 이슈 사이클 전체를 bootstrap한다.

초기 실행에서는 Q2가 28.0%를 바꾸는 정책을 골라 사전 10% 상한과 충돌했다.
이는 H2 점수에 따른 조정이 아니라 선택 단계의 동일 상한 누락이므로, Q2에도
10% 상한을 적용하는 계약 결함을 수정한 뒤 다시 실행했다.

## 최종 잠금 결과

선택 family는 `direction4`, 정책은 상향 행·출력비 20–80%·beta 1.0이었다.

| 지표 | H2 변화 |
|---|---:|
| 그룹 3 score | `+0.0006016738` |
| 그룹 3 1-NMAE | `+0.0000054198` |
| 그룹 3 FiCR | `+0.0011979279` |
| 변경 행 | `240 / 4,409` (`5.44%`) |
| 최대 추가 이동 | `210 kWh` (`1%` 용량) |

월별 score는 7·8·9·12월에 양수였지만 10월 `-0.0002527130`, 11월
`-0.0004148877`이었다. 이슈 블록 bootstrap은 다음과 같다.

- score q05 `-0.0000910729`
- 1-NMAE q05 `-0.0001112350`
- FiCR q05 `-0.0001732758`
- 세 구성지표 동시 양수 비율 `48.95%`

Q1 curve를 그대로 H2에 적용하는 정적 강건성 검사에서도 1-NMAE가
`-0.0000454064` 하락했다. 잠금 최소 그룹 3 score 개선 `+0.002`,
월별 비음수, bootstrap q05 양수 조건을 모두 만족하지 못했다.

## 판정과 다음 경계

- 판정: `rejected`
- CSV: 생성하지 않음
- 그룹 2 alpha/coverage 확대: 종료
- 그룹 3 풍향·시어별 isotonic curve: 종료

다음 실험은 같은 curve를 더 나누거나 H2의 10·11월을 사후 제외해서는 안 된다.
새 후보는 현재 KMA incumbent를 기준으로 6%·8% 경계 통과 확률을 직접
cross-fit하는 별도 목적함수이거나, 2023→2024를 포함하는 다기간
issue-block validation 자산을 먼저 구축한 뒤에만 진행한다.

기계 판정 보고서:
`artifacts_final/external_weather/kma_um_regional_context_2024/group3_regime_power_curve_20260726.json`
