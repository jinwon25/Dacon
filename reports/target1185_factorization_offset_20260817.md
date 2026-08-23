# 1185 목표 저랭크 FM 잔차 감사 — 2026-08-17

## 결론

초기 실험 레지스트리에 사전 등록됐지만 실행되지 않았던 factorization-machine
계열을 v27 위에서 누수 없이 검증했다. 평균 gain은 최근 신규 구조 중 가장 양호했지만
월별 안정성 gate를 통과하지 못해 2025 모델과 제출 ZIP은 만들지 않는다. champion은
`submit_v27.zip` / Public `1157.9736407889` 그대로다.

선택 레시피는 `rank=16 / R_CORE / logit damping=0.10`이었다. 2022와
late-2023의 평균 gain은 모두 양수였으나 2022 최악 월이 음수여서 사전 consensus
통과 후보는 0개였다. 선택에 쓰지 않은 2024에서도 평균은 양수였지만 월별 방향이
교대로 바뀌었다.

| 구간 | gain vs 해당 parent | 양의 월 비율 | 최악 월 |
|---|---:|---:|---:|
| 2022 선택 | +3.0979 | 75% 미만 | -3.8540 |
| late-2023 선택 | +12.3333 | 100% | +2.8491 |
| full-2024 진단 | +1.8216 | 50.0% | -10.8594 |
| late-2024 재현 | +5.5581 | 33.3% | -11.3971 |

full-2024의 적용 도메인 `R_CORE` gain은 `+2.5859`, late-2024는 `+7.9383`이었다.
그러나 후기 세 달은 8월 `-2.0004`, 9월 `+15.9389`, 10월 `-11.3971`로 한 달의
큰 이득이 두 달 손실을 가렸다. 목표인 안정적인 Top 10 후보의 근거로는 부족하다.

## 모델 구조와 중복 감사

기존 embedding MLP/TabM과 달리 hidden layer와 범주 주효과를 모두 제거했다. 다음
12개 사전 고정 상호작용만 rank 8 또는 16의 dot product로 표현했다.

- 투수×타자, 투수×카운트, 투수×손 조합, 투수×도메인
- 타자×손 조합, 타자×도메인, 투수팀×타자팀
- 카운트×손 조합/주자상태/압박, 도메인×압박, 이닝×압박

각 source OOF의 parent logit을 고정 offset으로 두고 binary log-loss를 학습했다.
source 도메인별 FM 보정 평균을 다음 origin에서 빼 전역 calibration 이동을 전달하지
않도록 했다. 따라서 이 실험은 기존 direct embedding 모델보다 저분산의 interaction
resolution만 겨냥한다.

## 시간축과 누수 통제

1. 2021 wave0 OOF 잔차로 2022를 예측했다.
2. 2022 prior-origin OOF 잔차로 late-2023을 예측했다.
3. rank·route·damping이 두 선택 원점에서 동일한 후보만 합의 대상으로 삼았다.
4. 선택 레시피를 고정한 뒤 late-2023 v27 OOF 잔차로 2024 모델을 한 번 학습했다.
5. 범주 어휘, 모델 파라미터, 도메인 중심값은 source 행만 사용했다. 평가 행의 빈도,
   다른 평가 행, target, 순서 정보는 사용하지 않았다.

2024 투수/타자 ID source-known 비율은 각각 `71.63%`, `86.26%`였다. 저랭크 공유로
cold-start 영향을 줄였지만, 선수 구성 변화와 월별 잔차 반전을 충분히 막지는 못했다.

## 결정과 재시도 금지

- rank 8/16, epoch 4, dropout 0.10, 동일 interaction set의 미세조정은 반복하지 않는다.
- 2024 결과를 보고 9월 전용 route 또는 10%보다 작은 damping을 고르지 않는다.
- 후기 평균 양수만 근거로 seed ensemble이나 2025 패키징을 진행하지 않는다.
- 재검토는 장·단기 source를 사전에 결합하는 별도 horizon 구조처럼 생성 가설이 달라질
  때만 허용한다. 동일 FM prediction의 사후 혼합은 허용하지 않는다.

## 재현

```powershell
python -m src.champion.v53_factorization_offset
python -m pytest tests/test_v53_factorization_offset.py -q
```

수치 산출물은 `artifacts/v53_factorization_offset_20260817_01/`에만 저장되며 Git에
포함하지 않는다. 사전 등록 근거는 `reports/top1100/experiment_registry.csv`의
`F1_fm_pilot_01/02`다.
