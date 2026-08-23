# 1185 목표 장·단기 FM 결합 감사 — 2026-08-17

> **2026-08-18 정정**: 이 보고서는 `submit_v27.zip`을 champion(frozen baseline)으로 가정하지만, 공식 DACON 제출 이력 재대조 결과 실제 champion은 `submit_v26.zip`이다(Public `1157.9736407889`, 제출 ID `51773`; v27은 `1156.6153781694`로 champion보다 낮다). 이 보고서의 v27 대비 gain 비교는 champion(v26) 기준으로 다시 확인하기 전까지 그대로 인용하지 않는다. 근거: [`target1170_followup_20260817.md`](target1170_followup_20260817.md) 상단, [`../notebooks/v26_champion_reproduction.ipynb`](../notebooks/v26_champion_reproduction.ipynb).

## 결론

사전 등록된 horizon-MoE 가설을 단순하고 누수 없는 형태로 실행했다. 두 직전 OOF
source에서 독립적으로 학습한 FM 보정의 균등 평균과 부호합의 평균을 검증한 결과,
v53 단기 FM보다 월별 변동은 크게 줄었지만 사전 선택 gate와 목표 최소 개선폭을 넘지
못했다. 2025 모델·제출 ZIP은 만들지 않으며 champion은 `submit_v27.zip`이다.

선택 레시피는 `rank=16 / sign-agreement / R_ANCHOR / logit damping=0.10`이었다.
2022와 late-2023 평균은 모두 양수였지만 각 구간 최악 월이 음수여서 64개 후보 중
consensus 통과 후보는 0개였다.

| 구간 | gain vs 해당 parent | 양의 active 월 비율 | 최악 active 월 |
|---|---:|---:|---:|
| 2022 선택 | +1.7032 | 기준 미달 | -1.3535 |
| late-2023 선택 | +0.8470 | 기준 미달 | -1.1637 |
| full-2024 진단 | +0.8611 | 71.4% | -0.9478 |
| late-2024 재현 | +2.1233 | 100% | +1.0031 |

full-2024의 R_ANCHOR active gain은 `+4.8887`, late-2024는 `+12.1439`였다. 다만
전체 gain `+0.8611`은 승격 최소값 `+5`에 훨씬 못 미치고 active 월 비율도 75%에
미달한다. 2024에서 R_ANCHOR 결과가 양수인 사실을 보고 damping을 키우는 것은 외부
감사 사후 선택이므로 실행하지 않는다.

## 설계

각 origin마다 서로 학습 정보를 공유하지 않는 두 FM을 만들었다.

- 2022 평가: 2020 OOF + 2021 OOF
- late-2023 평가: 2021 OOF + 2022 OOF
- 2024 평가: 2022 OOF + late-2023 v27 OOF

`mean` 전문가는 두 보정의 50/50 평균, `agree` 전문가는 부호가 같은 행에서만 같은
평균을 사용한다. rank 8/16, 두 전문가, `ALL/R_CORE/R_ANCHOR/F`, damping
`0.10/0.25/0.50/1.00`의 exact recipe가 2022와 late-2023에서 동일해야 했다.
평가행의 집계·순서·빈도는 사용하지 않았다.

## 해석

부호합의는 v53의 full-2024 최악 월을 `-10.8594 → -0.9478`, late-2024 최악
월을 `-11.3971 → +1.0031`로 줄였다. 대신 적용량도 평균 절대 확률 이동
`0.001857 → 0.000232`로 약 1/8이 되어 전체 개선폭이 작아졌다. 즉 장·단기 합의는
분산을 줄였지만 1185 격차를 줄일 충분한 resolution을 남기지 못했다.

또한 v53에서 FM family의 2024를 이미 확인했으므로 이번 2024는 family-level로
완전히 pristine한 감사가 아니다. 이 이유만으로도 작은 양수 결과를 제출 근거로
승격하지 않는다.

## 재시도 금지

- 50/50을 25/75 등으로 바꾸거나 부호 임계값을 미세조정하지 않는다.
- 2024 R_ANCHOR 양수 결과를 보고 route·damping을 다시 고르지 않는다.
- rank·epoch·dropout 또는 source 기간을 늘리는 동일 FM family 탐색을 종료한다.
- 새로운 독립 검증 연도나 새로운 관측 피처가 없으면 FM을 제출 후보로 재개하지 않는다.

## 재현

```powershell
python -m src.archive.v54_horizon_factorization
python -m pytest tests/test_v54_horizon_factorization.py -q
```

수치 산출물은 `artifacts/v54_horizon_factorization_20260817_01/`에만 저장하며 Git에
포함하지 않는다. 사전 등록 근거는 `reports/top1100/experiment_registry.csv`의
`F6_horizon_01`이다.
