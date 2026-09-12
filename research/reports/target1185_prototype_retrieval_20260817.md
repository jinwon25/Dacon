# 1185 목표 train-only prototype retrieval 감사 — 2026-08-17

## 결론

사전 등록된 prototype retrieval을 누수 없는 저용량 구조로 실행했지만 선택 단계부터
실패했다. 최선 레시피인 `R_ANCHOR / weight=0.25`도 2022와 late-2023에서 모두
음수였고 full-2024도 악화했다. 2025 모델·제출 ZIP은 만들지 않으며 champion은
`submit_v27.zip`이다.

| 구간 | gain vs 해당 parent | 양의 active 월 비율 | 최악 active 월 |
|---|---:|---:|---:|
| 2022 선택 | -0.7331 | 기준 미달 | -5.6439 |
| late-2023 선택 | -1.7029 | 기준 미달 | -3.4325 |
| full-2024 진단 | -0.6406 | 42.9% | -3.8397 |
| late-2024 재현 | +0.1103 | 50.0% | -0.5488 |

12개 route/damping 후보 중 consensus gate 통과 후보는 0개였다. full-2024의 적용
도메인 R_ANCHOR gain도 `-3.6369`였으며, late-2024의 작은 양수만으로 방법을
재선택하지 않는다.

## 구조

정확한 `domain × balls × strikes × pitcher hand × batter hand` 버킷 안에서 공식
행별 ASOF·경기상황 수치 27개를 robust scaling했다. source 행만 사용한
MiniBatchKMeans로 버킷당 최대 32개 프로토타입을 만들고, 각 프로토타입에는 다음
값을 저장했다.

`EB mean(source target - source OOF parent - source bucket residual mean)`

- 프로토타입 하나당 최소 목표 표본수: 256
- 잔차 smoothing: 200
- 검색: 같은 버킷의 가장 가까운 frozen 프로토타입 1개
- 선수 ID: 미사용
- 평가/test 행: 검색 reference로 미사용
- 미지 버킷: 정확히 0 backoff

실제 source 버킷은 모두 144개였고, 프로토타입 수는 2021 `831`, 2022 `848`,
late-2023 `371`개였다. 평가 버킷 coverage는 모든 origin에서 100%였다. 따라서
실패 원인은 cold bucket이 아니라 유사 상태 안의 잔차 방향이 시즌을 넘어 유지되지
않는 데 있다.

## 시간축과 누수 통제

1. 2021 wave0 OOF 잔차로 2022를 예측했다.
2. 2022 prior-origin OOF 잔차로 late-2023을 예측했다.
3. 같은 route와 damping이 두 선택 원점에서 통과해야 했다.
4. recipe를 고정한 뒤 late-2023 v27 OOF만으로 2024 프로토타입을 만들었다.
5. 평가 행의 좌표·빈도·거리 분포·target·다른 평가 행은 fitting에 들어가지 않았다.

프로토타입 좌표는 target-free다. target은 좌표가 고정된 뒤 source OOF 잔차 평균을
계산하는 데만 사용한다.

## 재시도 금지

- prototype 수, 최소 표본수, smoothing, 최근접 이웃 수의 미세 탐색을 하지 않는다.
- 2024 후기를 보고 R_ANCHOR에만 더 작은 weight를 사후 적용하지 않는다.
- 선수 ID 또는 현재 평가행을 retrieval reference에 추가하지 않는다.
- 동일 ASOF 상태공간의 TabR/k-NN 확장은 계산비용과 누수 위험만 늘리고 두 선택
  원점이 이미 음수이므로 중단한다.

## 재현

```powershell
python -m src.archive.v55_prototype_retrieval
python -m pytest tests/test_v55_prototype_retrieval.py -q
```

수치 산출물은 `artifacts/v55_prototype_retrieval_20260817_01/`에만 저장하며 Git에
포함하지 않는다. 사전 등록 근거는 `reports/top1100/experiment_registry.csv`의
`F5_retrieval_01`이다.
