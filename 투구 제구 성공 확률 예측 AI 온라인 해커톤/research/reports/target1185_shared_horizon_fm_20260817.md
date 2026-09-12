# 1185 목표 두-source 공유 FM 감사 — 2026-08-17

## 결론

두 연도의 FM을 따로 평균한 v54와 달리, 두 source OOF를 한 모델에서 공동 학습해
interaction embedding을 공유했다. 신호 크기는 조금 회복됐지만 2022 월 안정성 gate와
2024 최소 개선폭을 통과하지 못했다. 2025 모델·제출 ZIP은 만들지 않으며 champion은
`submit_v27.zip`이다.

선택 레시피는 `rank=16 / source×domain equal risk / F / logit damping=0.10`이었다.

| 구간 | gain vs 해당 parent | 양의 active 월 비율 | 최악 active 월 |
|---|---:|---:|---:|
| 2022 선택 | +0.1837 | 기준 미달 | -6.5616 |
| late-2023 선택 | +21.6487 | 100% | +22.5000 |
| full-2024 진단 | +1.0181 | 75.0% | -4.7126 |
| late-2024 재현 | +2.9166 | 100% | +0.3596 |

32개 exact risk/route/damping 후보 중 consensus gate 통과 후보는 0개였다.
full-2024 F active gain은 `+8.6512`, late-2024는 `+23.4828`이었지만 전체 gain은
승격 기준 `+5`에 훨씬 못 미친다.

## 구조

- 고정 rank 16, v53과 같은 12개 야구 저랭크 상호작용
- hidden layer·범주 주효과 없음
- 각 source 행의 OOF parent logit을 고정 offset으로 사용
- uniform row risk와 source-period×domain equal-total risk 두 개만 비교
- 두 source period/domain별 correction 평균을 계산한 뒤 period 균등 domain 평균 제거
- 평가행 어휘·빈도·집계·target 미사용

시간축은 다음과 같다.

1. 2020+2021 OOF 공동 학습 → 2022 선택
2. 2021+2022 OOF 공동 학습 → late-2023 선택
3. exact risk/route/damping 고정
4. 2022+late-2023 v27 OOF 공동 학습 → full/late-2024 감사

## 해석

별도 모델의 부호합의만 남긴 v54보다 shared representation은 late-2023과 2024 F에서
더 큰 resolution을 유지했다. 그러나 2022에서는 F 전체 gain이 거의 0이고 한 월은
`-6.56`이었다. 즉 공유 embedding도 2022→2023의 F 생성구조 단절을 해결하지 못했다.

또한 v53/v54에서 FM family의 2024 결과를 이미 확인했으므로 이번 2024는
family-level pristine audit가 아니다. 전체 gain `+1.02`를 근거로 F damping을 키우거나
epoch·rank를 조정하면 반복 사용한 2024에 과적합된다.

## 재시도 금지

- shared FM의 rank, epoch, dropout, source 위험 가중치 미세조정을 종료한다.
- full-2024 F active gain을 보고 damping을 10%보다 키우지 않는다.
- FM 단독·평균·부호합의·공유 학습 계열을 새로운 독립 연도 없이 재개하지 않는다.

## 재현

```powershell
python -m src.champion.v56_shared_horizon_fm
python -m pytest tests/test_v56_shared_horizon_fm.py -q
```

수치 산출물은 `artifacts/v56_shared_horizon_fm_20260817_01/`에만 저장하며 Git에
포함하지 않는다.
