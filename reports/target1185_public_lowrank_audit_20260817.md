# 1185 목표 공개 코드·저랭크 상호작용 감사 — 2026-08-17

> **2026-08-18 정정**: 이 보고서는 `submit_v27.zip`을 champion(frozen baseline)으로 가정하지만, 공식 DACON 제출 이력 재대조 결과 실제 champion은 `submit_v26.zip`이다(Public `1157.9736407889`, 제출 ID `51773`; v27은 `1156.6153781694`로 champion보다 낮다). 이 보고서의 v27 대비 gain 비교는 champion(v26) 기준으로 다시 확인하기 전까지 그대로 인용하지 않는다. 근거: [`target1170_followup_20260817.md`](target1170_followup_20260817.md) 상단, [`../notebooks/v26_champion_reproduction.ipynb`](../notebooks/v26_champion_reproduction.ipynb).

## 결론

현재 제출 우선순위는 바꾸지 않는다. champion은 계속 `submit_v27.zip`, Public
`1157.9736407889`이다. 공개 저장소에서 독립적으로 확인한 투수×카운트×타자 손
저랭크 SVD 신호를 현 계보 위에서 다시 검증했지만 제출 안정성 gate를 통과하지
못했다. 제출 ZIP과 2025 전체 학습 모델은 만들지 않았다.

| 구간 | gain vs 해당 v27 analogue | 양의 월 비율 | 최악 월 | R_CORE | R_ANCHOR | F |
|---|---:|---:|---:|---:|---:|---:|
| 2022 선택 | +7.5280 | 100.0% | +2.4633 | — | — | — |
| late-2023 선택 | +10.6956 | 100.0% | +5.5537 | — | — | — |
| full-2024 외부 감사 | **+2.1194** | 37.5% | -10.9248 | -0.1360 | +1.2869 | +16.8970 |
| late-2024 재현 | **+0.1128** | 33.3% | -3.0186 | -3.1898 | +0.8744 | +17.6592 |

평균 gain의 부호는 네 축에서 모두 양수였지만 full-2024 최소 요구치 `+5`, 양의 월
비율, 최악 월, 최소 도메인 조건을 충족하지 못했다. 특히 F에서는 강한 반면
R_CORE에서 반복적으로 음수였다. 2024 결과를 본 뒤 F route나 더 작은 weight로
바꾸는 것은 사후 선택이므로 금지한다.

## 공개 자산 감사

두 공개 저장소를 읽기 전용으로 확인했다.

- `danny010712/LG-Aimers-9th-Hackathon`은 외부 KBO 자료로 추정한 2025 성공률
  `0.477`에 맞춘 고정 logit shift가 Public을 약 `950.96 -> 998.00`으로 올랐다고
  기록한다. 이 상수는 제공 데이터만으로 정해지지 않았고 Public 결과까지 함께
  알려진 상태이므로 현 프로젝트의 모델 파라미터로 복사하지 않았다.
- 이 저장소에는 이미 `src/calibration.py`의 train-only 계절 base-rate 예측과
  `damped_3_0.8` 검증이 있으며 v27 계보도 계절 추세 보정을 포함한다. 따라서
  공개 global shift는 규정 위험뿐 아니라 기존 구성과도 상당 부분 중복된다.
- `mk-isos/lg-aimers-9-pitch-control`의 EXP-020은 각 과거 OOF 시즌 잔차를
  투수×24개 count/타자 손 행렬로 만들고 EB shrinkage 뒤 truncated SVD로
  복원했다. 공개 기록상 해당 저장소의 rolling 2022/2023/2024 개선은 각각
  `+5.20/+8.93/+19.52`였고 Public도 이전 계보보다 크게 올랐다. 현 저장소에는
  계층 EB와 선수 embedding은 있었지만 이 정확한 저랭크 행렬 구조는 없어서
  독립 후보로 채택해 재구현했다.

참고:

- <https://github.com/danny010712/LG-Aimers-9th-Hackathon>
- <https://github.com/mk-isos/lg-aimers-9-pitch-control>

## v50 프로토콜

1. source season `2020..2023` 각각에서 train-only wave0 OOF의
   `target - prediction` 잔차를 구하고 시즌 평균을 제거한다.
2. `pitcher_id × (balls, strikes, batter_hand)` 24셀 합/수를 만든다.
3. smoothing `300/600`, rank `2/4/6`만 미리 선언하고 결정론적 SVD로 복원한다.
4. audit year보다 엄격히 앞선 source season 복원값을 동일 가중 평균한다. 특정
   source season에서 보지 못한 투수는 0을 기여한다.
5. smoothing/rank/domain/weight가 완전히 같은 레시피만 2022와 late-2023에서
   비교한다. 두 축 모두 평균·월·적용 도메인이 양수인 42/144개만 통과했다.
6. 최대 최소 gain 규칙으로 `s300/r2`, `ALL`, weight `0.50`을 고정한 뒤에만
   full-2024와 late-2024를 열었다.

2024 행의 과거 투수 coverage는 any-source `80.14%`, every-source `35.90%`였다.
저랭크 복원은 관측하지 않은 정확한 count/hand cell에도 값을 공유하지만, 여러
시즌에 연속 재직한 투수의 비율은 낮아 오래된 source를 동일 가중할수록 신호가
희석된다. 반대로 최근 source를 2024 결과에 맞춰 더 크게 두는 재탐색은 하지
않는다.

## 실패 원인과 재시도 금지

- 이 신호는 투수별 문맥 효과 자체는 잡지만 정규시즌의 월별 잔차 방향을 안정적으로
  예측하지 못했다.
- 2024에서 좋은 F 성적만 보고 `F-only`로 바꾸지 않는다. 사전 선택 단계에서
  F-only 통과 후보는 0개였다.
- 동일 SVD grid의 rank/smoothing/weight 미세조정, 2024 우수 레시피 선택,
  최근 source 가중치 사후 변경은 재시도하지 않는다.
- 다음 후보는 SVD와 다른 정보인 공식 행별 `asof_pitcher_*`에서 현재 시즌 표본을
  복원하고, 과거 시즌 투수 상태의 train-only AR(1) prior를 결합하는 저자유도
  동적 상태 보정이다. 이것도 동일한 두-origin 선택과 2024 외부 감사를 적용한다.

## 재현

```powershell
python -m src.v50_low_rank_pitcher_context
python -m pytest tests/test_v50_low_rank_pitcher_context.py -q
```

수치 산출물은 `artifacts/v50_low_rank_pitcher_context_20260817_01/`에만 저장되며
Git에 포함하지 않는다.
