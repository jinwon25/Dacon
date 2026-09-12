# 비교차 분포 코어 기준선 대칭성 감사 — 2026-07-29

## 최종 결론

직전 비교차 분포 실험의 모델 아이디어가 아니라 검증 계약에서 오류를
발견했다. 2023 선택은 `분포 중앙값 → 분포 Bayes action`을 비교했지만,
2024 확인은 같은 action delta를 별도 incumbent 위에 더했다. FiCR는 예측값이
실제값의 6%·8% 오차 절벽 중 어디에 위치하는지에 따라 효용이 달라지므로,
서로 다른 기준선에 같은 delta를 적용한 두 결과는 전진 전이라고 볼 수 없다.

실험을 다음 두 비교로 분리했다.

1. **family-internal transfer**: 2023과 2024 모두 분포 중앙값을 기준으로
   동일한 Bayes action 보간을 평가한다.
2. **incumbent replacement**: 완성된 분포 코어 예측 전체를 정확한 2024
   incumbent OOF와 비교한다.

incumbent에 action delta만 얹는 비대칭 overlay는 코드의 승격 경로에서
제거했다. 새 제출 CSV는 생성하지 않았으며 incumbent `1502437`을 유지한다.

## 수정한 계약

- 학습: 2022 → 2023 선택, 2023 → 2024 확인
- 선택 가중치: `0.50`
- 선택 기준선: 2023 분포 코어 중앙값
- 내부 확인 기준선: 2024 분포 코어 중앙값
- 배포 비교: 완성된 2024 분포 후보 대 정확한 active incumbent OOF
- 금지: 분포 중앙값에서 선택한 delta를 incumbent에 직접 overlay
- 공통 gate: 전체·H2·월별·IID 40/60·월층화 40/60·이동량
- 제출 writer: 없음

## 2023 선택 결과

G1/G2 affected-pair macro delta:

| 구성요소 | delta |
|---|---:|
| score | `+0.0080624533` |
| 1-NMAE | `+0.0034886831` |
| FiCR | `+0.0126362235` |

IID 40% q05는 score `+0.0059515670`, 1-NMAE `+0.0031616366`,
FiCR `+0.0085899734`였다. 보완 60% q05도 모두 양수여서 2024 확인은
정상적으로 열렸다.

## 2024 family-internal transfer

G1/G2 affected-pair macro delta:

| 구성요소 | 전체 | H2 |
|---|---:|---:|
| score | `+0.0104015838` | `+0.0094997204` |
| 1-NMAE | `+0.0028722084` | `+0.0027499066` |
| FiCR | `+0.0179309593` | `+0.0162495342` |

IID 40% q05:

- score `+0.0080114680`
- 1-NMAE `+0.0025292757`
- FiCR `+0.0132373041`

보완 60% q05:

- score `+0.0088122123`
- 1-NMAE `+0.0026420465`
- FiCR `+0.0148417836`

즉 분포에서 유도한 행동 자체는 2023에서 2024로 강하게 전이됐다. 이전
보고서에서 관찰한 2024 1-NMAE 음수는 행동 비대칭의 연도 이동이 아니라
기준선 혼용에서 발생했다.

다만 7월 score가 `-0.0076969770`, 1-NMAE가 `-0.0038480910`,
FiCR가 `-0.0115458630`이어서 every-month gate는 실패했다.

## 2024 incumbent 전체 교체

완성된 분포 후보를 incumbent와 직접 비교한 affected-pair macro delta:

| 구성요소 | 전체 | H2 |
|---|---:|---:|
| score | `-0.0280773931` | `-0.0300938299` |
| 1-NMAE | `-0.0158115263` | `-0.0169490190` |
| FiCR | `-0.0403432600` | `-0.0432386408` |

IID·월층화 40/60의 모든 score q05가 음수이고 양수 비율은 `0%`다.
incumbent 대비 평균 절대 차이는 용량의 `6.43%`, p95는 `15.49%`,
최댓값은 `35.70%`였다. 작은 안전 보정이 아니라 훨씬 약한 standalone
코어로의 대규모 교체이므로 승격할 수 없다.

## 정정된 병목

이전 진단:

- 조건부 FiCR 행동이 연도 사이에서 반전한다.

대칭성 감사 후 진단:

- 조건부 행동은 실제로 두 전진 연도에서 모두 개선됐다.
- 실패 원인은 비교차 MLP의 절대 중앙 예측이 현재 incumbent 앙상블보다
  크게 약한 것이다.
- 행동 delta만 incumbent에 옮기면 선택과 배포의 기준선이 달라져
  공식 효용의 전진 검증이 깨진다.

따라서 다음 돌파구는 행동 폭이나 가중치 재조정이 아니다. **incumbent와
동일한 causal 기준선을 과거 연도에도 만들 수 있는 residual-distribution
코어**, 또는 incumbent 자체가 출력하는 다양한 OOF 멤버를 입력으로
사용하는 baseline-matched 분포 모델이 필요하다.

현재 G3는 2022 라벨이 없어 동일 incumbent의 2023 OOF를 만들 수 없다.
G1/G2만 별도 causal baseline을 재구축할 수 있지만, 그 baseline으로 선택한
delta를 현재 incumbent에 옮기는 것도 동일한 문제가 있으므로 완성 예측
전체가 incumbent를 이겨야 한다.

## 코드 안전장치

`experiments/noncrossing_distributional_core.py`에 다음을 고정했다.

- `interpolate_distribution_action`: 항상 자체 median을 anchor로 사용
- `family_internal`과 `incumbent_replacement` 결과 분리
- `baseline_symmetry_required=true`
- `incumbent_delta_overlay_allowed=false`
- 두 비교가 모두 qualified일 때만 최종 qualified
- 제출 파일 writer 없음

단위 테스트는 자체 median anchor와 행동 상한을 검증한다.

## 산출물

- `experiments/noncrossing_distributional_core.py`
- `tests/test_noncrossing_distributional_core.py`
- `artifacts_final/diagnostics/noncrossing_distributional_core_symmetric_20260729.json`
- `artifacts_final/diagnostics/noncrossing_distributional_core_symmetric_smoke_20260729.json`

