# 1185 목표 시간순 convex OOF stacking 감사 — 2026-08-17

> **2026-08-18 정정**: 이 보고서는 `submit_v27.zip`을 champion(frozen baseline)으로 가정하지만, 공식 DACON 제출 이력 재대조 결과 실제 champion은 `submit_v26.zip`이다(Public `1157.9736407889`, 제출 ID `51773`; v27은 `1156.6153781694`로 champion보다 낮다). 이 보고서의 v27 대비 gain 비교는 champion(v26) 기준으로 다시 확인하기 전까지 그대로 인용하지 않는다. 근거: [`target1170_followup_20260817.md`](target1170_followup_20260817.md) 상단, [`../notebooks/v26_champion_reproduction.ipynb`](../notebooks/v26_champion_reproduction.ipynb).

## 결론

새 제출 후보를 만들지 않았다. 현재 champion은 계속 `submit_v27.zip`, Public
**1157.9736407889**다. 기존 실험에서 개별 OOF 신호와 최대 2개 조합은 광범위하게
검사했지만, 여러 생성 계열을 함께 학습하는 meta-model은 빠져 있었다. 이를 보완한
v49 convex stack은 late-2023에서 개선됐으나 full/late-2024에서 모두 반전해
기각했다.

| 구간 | gain vs v27 analogue | 양수 월 비율 | 최악 월 | 적용 F gain |
|---|---:|---:|---:|---:|
| late-2023 선택 | **+6.0867** | 100.0% | +5.9784 | +70.3903 |
| full-2024 감사 | **-1.7446** | 37.5% | -7.6862 | -14.8246 |
| late-2024 재현 | **-1.9833** | 33.3% | -2.1932 | -15.9689 |

수치 게이트를 통과하지 못했으며, 2024를 본 뒤 부호·route·가중치를 바꾸지 않았다.

## 독립 자산 재감사

로컬 저장소와 기존 전달물을 다시 확인했지만 v27 계보 밖에서 새로 생성된 팀원
row-level OOF와 2025 test prediction은 없었다. 기존 48개 공통 OOF 열은 다음 네
생성 계열이다.

- `exact`: exact-ASOF 기반 직접모형
- `recent`: 최근 시즌 exact-ASOF 모형
- `state`: 다년 current-season state 모형
- `mode`: 잠재 실패유형 모형

이들은 개별·소수 조합으로는 이미 검사됐지만, convex meta-model 자체는 이번에
처음 시간순으로 평가했다.

## v49 프로토콜

1. 2022 season-forward OOF 48열과 2022 target으로 non-negative simplex stack을
   학습했다. 계수는 합이 1이며 음수 가중치를 금지했다.
2. 48열 직접 stack과 4개 family 평균 stack, ridge-to-uniform 강도
   `0/1e-4/1e-3/1e-2`, 적용 domain, v27 혼합비를 late-2023에서만 선택했다.
3. 선택 점수는 전체 gain, 최악 active month gain, **실제 적용 domain gain**의
   최소값이다. 적용하지 않은 domain의 0을 선택 점수에 넣지 않는다.
4. 선택된 meta recipe를 full-2023 OOF로 재학습하고 full-2024에 한 번 적용했다.
   late-2024는 같은 예측의 보조 안정성 감사다.

최종 선택은 family 평균, regularization `0.01`, F-only, v27에서 직접 stack 쪽으로
20% 이동이다. 320개 선택 후보 중 54개가 late-2023 gate를 통과했다.

## 계수 안정성과 반전 원인

| family | 2022 학습 계수 | 2023 재학습 계수 |
|---|---:|---:|
| exact | 0.2556 | 0.2188 |
| mode | 0.2361 | 0.2739 |
| recent | 0.2487 | 0.2730 |
| state | 0.2595 | 0.2343 |

계수 L1 이동은 `0.1240`, 단일 최대 이동은 `0.0378`로 작다. 특정 계열 하나가
붕괴해서 실패한 것이 아니라, F에서 v27 대비 direct stack 방향 자체가 시점마다
바뀌었다.

| 구간(F만) | target 평균 | v27 평균 | direct 평균 | `E[(y-p)d]` | 무제약 최적 eta |
|---|---:|---:|---:|---:|---:|
| late-2023 | 0.46216 | 0.62063 | 0.61813 | +0.00046184 | +1.891 |
| full-2024 | 0.45928 | 0.45771 | 0.46230 | -0.00007167 | -0.352 |
| late-2024 | 0.47043 | 0.46038 | 0.45735 | -0.00008460 | -0.570 |

여기서 `p`는 v27, `d=direct-p`다. late-2023에서는 두 모델 모두 F 성공률을 크게
과대예측했고 direct stack이 아주 조금 낮아지는 방향이라 개선됐다. 2024에서는
v27의 F 평균이 target에 이미 가까워졌고 direct 방향과 v27 잔차의 내적이 음수로
반전했다. late-2024에서는 v27이 오히려 낮은데 direct가 더 낮아져 같은 부호로
악화됐다. 따라서 작은 eta 재탐색으로 해결할 문제가 아니다.

초기 진단 코드에서 route 밖 domain의 0을 최소 domain 값으로 사용해 후보들이
동률이 되는 문제를 발견했다. 최종 프로토콜에서는 실제 적용 domain gain으로
수정했다. 다만 그 과정에서 같은 base family의 2024 결과를 한 번 열었으므로,
코드와 보고서에 `family_reused_audit_risk=true`를 남겼다. 최종 선택도 2024를
통과하지 못해 승격 판단에는 영향이 없다.

## 누수·규칙 감사

- 모든 meta 입력은 각 audit season의 season-forward OOF 예측이다.
- stack 학습 target은 해당 source OOF 시즌에만 사용했다.
- model/regularization/route/eta는 late-2023에서 고정했고 2024 label은 선택에
  사용하지 않았다.
- 결과는 현재 행의 base prediction만 convex 결합하며 다른 평가 행의 값·빈도·순서나
  test 전체 분포를 사용하지 않는다.
- 2024를 본 뒤 음수 계수, 역방향 혼합, F 제외, 월별 route를 선택하지 않았다.

## 재시도 금지

새 독립 OOF가 없으면 다음을 반복하지 않는다.

1. 같은 48개 열의 simplex/ridge/NNLS regularization 미세조정
2. late-2023 F gain을 이용한 혼합비 확대·축소
3. 2024 반전을 본 뒤 negative weight 또는 `v27-direct` 역방향 선택
4. 48개 열 중 2024 성능이 좋은 일부만 사후 선택
5. 같은 네 family를 다른 이름으로 재평균한 stack

## 다음 독립 예측 intake 조건

다음으로 가치가 있는 입력은 현재 네 family와 별도로 학습된 row-level 예측이다.
받을 때 아래 항목을 고정 검증한다.

- `model_id`, 코드/config hash, 학습 cutoff와 fold 정의
- 원본 `row_id`, audit season, target, OOF prediction의 1:1 정렬
- 각 OOF 행이 자기 target을 학습한 모델에서 나오지 않았다는 provenance
- 2022·2023·2024 공통 prediction과 동일 recipe의 2025 test prediction
- test 행 독립성, 네트워크·외부 데이터·test 집계 미사용
- v27과의 예측 상관뿐 아니라 `error × direction` 공분산과 월·domain 안정성

이 조건을 만족하는 새 자산 전에는 제출 우선순위를 `v27`에서 바꾸지 않는다.

## 재현

```powershell
python -m src.v49_temporal_convex_stack --project . --output-dir artifacts/v49_temporal_convex_stack_20260817_01
python -m pytest -q
```

OOF와 예측 산출물은 `artifacts/v49_temporal_convex_stack_20260817_01/`에만 저장되며
Git에 포함하지 않는다.
