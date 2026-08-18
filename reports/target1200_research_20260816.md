# Target 1200 심층 연구 및 v21 의사결정 — 2026-08-16

## 결론

1200은 현재 리더보드 3위권 수준이며, v20에서 필요한 증분은 `+48.5276`이었다. 단일 제출권으로 감당할 수 있는 불확실성을 고려해 1200 자체를 제출 게이트로 삼지 않고, v20 대비 세 주요 시간축에서 모두 양수이고 군집 부트스트랩 하단도 양수인 후보만 제출하기로 했다.

최종 v21은 이 게이트를 통과했다. 다만 Public 개선은 `+0.0414`로 로컬 `+10.7720`보다 크게 축소됐다. 따라서 점수 방향은 맞았지만, 1200 또는 Top 10을 설명할 정도의 새 일반화 축은 찾지 못했다.

## 기준선 진단

- v19 → v20 로컬 2024 gain: `+14.2517`
- v19 → v20 Public gain: `+7.3206`
- 관측 전이율: `0.5137`
- v20 월별 gain: 8개 월 중 5개 양수
- v20은 전체 보정 강도 scale `1.0107` 부근이 로컬 최적이어서 단순 증폭 여지가 거의 없었다.

## 검토한 방법과 결과

| 축 | 핵심 결과 | 판단 |
|---|---|---|
| 확률 shrink/stretch, beta형 보정 | 최대 약 `+0.56` | 제출 근거 부족 |
| v20 전체 보정 scale | 현재 scale이 사실상 최적 | 기각 |
| 도메인별 scale | 세 축 최소 약 `+0.45` | 기각 |
| 5개 v20 성분 재최적화 | 다음 시점에서 최대 `-278` | 선택 과적합 |
| nested state/mode routing | 2024에서 v19 대비 `-53` 이상 | regime 실패 |
| 추가 failure-mode/state 신호 | 최소 gain 약 `+0.7~1.0` | 불충분 |
| latent pitch type | oracle는 크지만 추론 불가; 합법 student는 미미 | 미채택 |
| batter current-season 재구성 | 2022→23 방향이 2023→24에서 반전 | 불안정 |
| 실패유형 EB | 2022→23 대폭 개선, 2023→24 거의 0 | regime 실패 |
| exact-state cohort | 최소 약 `+0.2` | 불충분 |
| CatBoost failure-mode classifier | 2023 `+108`, 2024 `+1.83` | 강한 시점 붕괴 |
| 기존 OOF 185개 메타 라이브러리 | 합법 개별 모델 최소 최대 `+1.71` | 불충분 |
| 기존 recency EB 재앙상블 | v20 대비 세 축 최소 `+3.29` | 유망, 추가 탐색 |
| 야구 맥락·최근 제구 상태 EB | 최종 세 축 최소 `+10.77` | v21 승격 |

현재 실패유형을 직접 사용하는 oracle은 매우 큰 gain을 보였지만 이는 추론 시점에 알 수 없는 현재 투구 결과 정보를 포함하므로 후보에서 제외했다. 현재 구종 oracle 역시 같은 이유로 제외했다.

## v21 신호

모든 효과는 2024 `target - v19 OOF` 잔차에서 학습하고 월 단위 지수 감쇠와 empirical-Bayes 분모를 적용해 동결했다.

| 신호 | 도메인 | 반감기 | alpha | weight |
|---|---|---:|---:|---:|
| 투수×타자손×최근 3경기 제구 10구간 | R_CORE | 0.5개월 | 200 | 0.275 |
| 투수×타자손×최근 5경기 제구 10구간 | R_CORE | 1개월 | 100 | 0.075 |
| 투수×타자손×주자상황 | 전체 | 1개월 | 25 | 0.050 |
| 카운트×투수손×타자손×이닝구간 | 전체 | 2개월 | 25 | 0.100 |
| 카운트×손조합×최근 3경기 제구 20구간 | 전체 | 0.5개월 | 800 | 0.200 |
| 카운트×손조합×reverse rate 10구간 | 전체 | 2개월 | 100 | 0.050 |

## 강건성 감사

| 검증 | simple4 | robust6(v21) |
|---|---:|---:|
| 2023 전체 → 2024 전체 | +10.1358 | **+10.7720** |
| 2023 7월까지 → 이후 | +47.8394 | **+50.6345** |
| 2024 7월까지 → 이후 | +11.7302 | **+14.1480** |
| 2024 6월까지 → 이후 | +3.2132 | **+4.4233** |
| 2024 5월까지 → 이후 | -12.0918 | **-13.8581** |

초기 3개월만 학습한 2024 rolling origin은 음수였다. 이 경고 때문에 최적화 해를 그대로 사용하지 않고 가중치를 단순 반올림했으며, 최종 판단은 전체 2023→2024 전이와 충분한 표본이 쌓인 최신 후반기 전이에 더 큰 비중을 뒀다.

## Public 결과가 주는 힌트

v21의 2024 전체 로컬 gain `+10.7720` 중 Public으로 전이된 값은 `+0.0414`뿐이었다. 추정 전이율 약 `0.00384`는 다음을 시사한다.

1. 2024 잔차 lookup을 더 세분화하는 방식은 이미 포화됐다.
2. 같은 source year에서 파생된 월·도메인 양수와 bootstrap 양수만으로 2025 구조적 이동을 보장할 수 없다.
3. 다음 큰 개선은 lookup 추가보다 2025에서도 유지되는 생성 구조, 특히 현재 시즌 상태의 정확한 베이지안 갱신이나 새로운 합법적 privileged distillation 표현에서 나와야 한다.
4. 향후 제출권이 생겨도 v21 계열의 미세 가중치 조정은 하지 않는다.

## 참고 문헌과 공식 자료

- DACON 평가식: https://dacon.io/competitions/official/236743/overview/evaluation
- DACON 규칙: https://dacon.io/competitions/official/236743/overview/rules
- DACON 독립 예측 안내: https://dacon.io/competitions/official/236743/talkboard/417123
- DACON TrackMan/ASOF FAQ: https://dacon.io/competitions/official/236743/talkboard/417082
- Kumar et al., calibrated ensembles under shift: https://proceedings.mlr.press/v180/kumar22a.html
- Han et al., model selection under temporal shift: https://proceedings.mlr.press/v235/han24b.html
- Kull et al., beta calibration: https://proceedings.mlr.press/v54/kull17a.html
- Park et al., calibration under covariate shift: https://proceedings.mlr.press/v108/park20b.html
- Lopez-Paz et al., generalized distillation: https://arxiv.org/abs/1511.03643
- Cawley & Talbot, selection overfitting: https://www.jmlr.org/papers/v11/cawley10a.html
- Owen, crossed-effects bootstrap: https://doi.org/10.1214/07-AOAS122
- xCTRL의 확률적 투구 위치 정확도 관점: https://wsb.wharton.upenn.edu/introducing-xctrl-a-probabilistic-approach-to-pitch-location-accuracy/
