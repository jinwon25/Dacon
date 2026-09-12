# 다기간 OOF 타당성·Temporal Smoothing·산출물 정리 — 2026-07-26

## 결론

다음 단계로 제안했던 `2023+2024 exact incumbent OOF`를 먼저 감사했다.
그러나 2022년 그룹 3 라벨이 전혀 없고 현재 cross-group lineage가 2023
그룹 3 라벨을 학습에 사용하므로, 2023 그룹 3을 과거 정보만으로 예측한
동일 incumbent OOF는 만들 수 없다. 미래 라벨을 사용한 값을 exact라고
기록하지 않았고, 이를 위해 KMA 2023 원시 예보를 새로 수집하지도 않았다.

대체 계약으로 2024 Q1 학습, Q2 전체 및 4·5·6월별 선택, H1 재학습 후
H2 단회 평가를 사용해 LDAPS hub-height wind temporal smoothing을
검증했다. Q2에서 세 구성지표를 모두 개선한 정책이 0개여서 H2를 열지
않고 종료했다. 제출 CSV는 생성하지 않았다.

## 2023 exact OOF 타당성

`data/train/train_labels.csv`의 연도별 non-null 행 수는 다음과 같다.

| 연도 | 그룹 1 | 그룹 2 | 그룹 3 |
|---:|---:|---:|---:|
| 2022 | 8,664 | 8,664 | 0 |
| 2023 | 8,757 | 8,758 | 8,759 |
| 2024 | 8,778 | 8,778 | 8,778 |

2023 그룹 3 fold를 causal하게 만들려면 2022 또는 그 이전 그룹 3 라벨로
학습해야 하지만 사용 가능한 행이 없다. 현재 그룹 3 cross-group 모델은
2023 실측 관계로 2024를 예측하므로 같은 모델로 2023을 평가하면 명백한
target leakage다. 현재 KMA power curve도 2024 그룹 3 라벨로 학습되므로
2023에 역적용한 값은 OOF가 아니다.

따라서 다음 두 표현은 사용하지 않는다.

- `2023 exact incumbent OOF`
- `2023→2024 causal transfer validation`

2023 KMA 원시 자료 수집만 추가해도 이 라벨 부재는 해결되지 않으므로,
약 1,468개 API 객체를 새로 만드는 작업은 생략했다.

## Temporal wind smoothing

구현:
`experiments/ldaps_temporal_wind_smoothing.py`

근거가 된
[Kanninen et al. (2021)](https://wes.copernicus.org/articles/6/1205/2021/)의
방법과 같이, 각 예보 발행 주기 안에서 `t-1, t, t+1` hub-height wind의
단순 moving average를 사용했다. 발행 주기 경계를 넘지 않으며 첫·마지막
lead는 자기 값을 복제해 채웠다.

### 실험 계약

- Q1: 2024-01-01–2024-03-31 power curve 학습
- Q2: 2024-04-01–2024-07-01 00:00 희소 정책 선택
- Q2 추가 게이트: 4·5·6월 각각 score, 1-NMAE, FICR 비음수
- 변경 행 상한: 10%
- 행별 추가 이동 상한: 용량의 1%
- H2: Q2를 통과한 경우에만 H1 재학습 후 단 한 번 평가

원본과 smoothing wind에 별도 monotone isotonic curve를 적합하고,
`smoothed power - raw power` 차이만 incumbent에 더했다. 따라서 새
power curve의 전체 편향을 주입하지 않는다.

### 결과

- 검토 정책: 48개
- Q2 세 구성지표 동시 양수: 0개
- Q2 월별 견고성 통과: 0개
- H2 평가: 미실행
- CSV: 미생성

총점 기준 최상 정책은 상향, disagreement 상위 10%, incumbent 20%–80%,
alpha 0.25였다.

| 지표 | Q2 변화 |
|---|---:|
| score | `+0.0000109559` |
| 1-NMAE | `-0.0000646197` |
| FICR | `+0.0000865316` |

월별 score는 4월 `-0.0003367192`, 5월 `-0.0013949366`, 6월
`+0.0034274745`였다. 평균적으로 좋아진 것이 아니라 6월 이득이 앞선 두
달 손실을 상쇄한 구조이며, NMAE는 세 달 모두 하락했다.

또한 기존 KMA OOF 캐시의 첫 경계 행에서 issue time `NaT` 1건을
발견했다. 알려진 8,778개 issue time은 LDAPS 원본과 오차 0시간으로
일치했고, 누락 1건만 LDAPS 원본에서 보완해 smoothing을 수행했다.

기계 판정:
`artifacts_final/diagnostics/ldaps_temporal_wind_smoothing_20260726.json`

## 산출물 정리 정책

원본 데이터, 현재 최고 제출, 제출 이력 CSV, 외부자료 manifest/raw,
exact lineage, 공유 feature cache, compact 진단 JSON과 재현 코드는
보존한다.

정리 대상은 다음으로 제한한다.

- Python bytecode 및 pytest cache: 언제든 재생성 가능
- `artifacts_final/logs/`: 보고서와 코드에서 참조하지 않는 실행 stdout/stderr
- `smoke_incumbent_blend_report.json`: 정식 결과로 대체된 smoke report
- `artifacts_final/candidates/`의 공개 제출 완료·비선택 CSV 4개:
  삭제하지 않고 `submissions/archive/`로 이동

현재 최고
`artifacts_final/candidates/kma_group2_overlay_alpha2375_20260725.csv`는
후보 디렉터리에 단독으로 유지한다. 이동·삭제 경로와 SHA-256은
`artifacts_final/cleanup_manifest_20260726.json`에 기록한다.
