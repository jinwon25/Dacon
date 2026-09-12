# 1185 목표 구조적 상태모형 후속 연구 — 2026-08-17

## 결론

`submit_v27.zip`과 Public **1157.9736407889**를 유지한다. 이번 사이클의
v39~v41은 잔차 lookup 미세조정이 아니라 현재 시즌 선수 상태와 잠재 투구
상태의 생성 구조를 새로 만들었지만, full-2024 승격 기준을 충족하지 못했다.
제출 ZIP을 만들거나 Public 점수를 확인하지 않았다.

| 버전 | 구조 | 선택 원점 | full-2024 | late-2024 | 판정 |
|---|---|---:|---:|---:|---|
| v39 | 직전 시즌·경력 prior로 수축한 당해 시즌 ASOF posterior | 2022 `+17.40`, late-2023 `+11.72` | `-2.58` | `+5.49` | 기각 |
| v40 | 2021 공식 사전선별 + pitcher season-n 신뢰도 게이트 | v39와 동일, `min_n=0` | `-2.58` | `+5.49` | 가설 반박 |
| v41 | 구종 3종 × 실패유형 4종 공동 잠재상태 학생 | 합의 gate 0개 | fallback `+0.119` | `+0.137` | 기각 |

## 완전 champion analogue OOF 재감사

과거 OOF 파일이 없다는 사실만으로 재생성을 포기하지 않고 v19→v27 코드
계보를 다시 추적했다. 원시 2019~2021 데이터에서 개별 base/state/mode 예측을
새로 만드는 것은 가능하지만, v19의 도메인 route와 배율 자체가 2023·2024
라벨을 함께 사용해 선택됐다. 그 고정값을 2021·2022에 적용하면 예측 행은
순방향이어도 **부모 선택은 미래를 본 것**이 된다.

따라서 이런 산출물을 “완전 2022 champion OOF”라고 부르지 않았고, 미래 선택
누수가 없는 저자유도 원자료 생성모형을 별도로 만들었다. 2022에서는 기존
부모를 v27 analogue가 아니라 방향 합의용 older OOF로만 사용했다.

## v39 — hierarchical current-season posterior

각 평가 행의 공식 ASOF 충분통계에서 이전 시즌의 labelled 누적 횟수와 성공
수를 빼 당해 시즌 `n/s`를 정확히 복원했다. 이 행별 증거를 다음 prior로
수축했다.

- 선수 전체 경력 prior
- 직전 시즌 선수 prior
- 경력 35% + 직전 시즌 65% prior
- 전체 또는 `R_CORE/R_ANCHOR/F` 도메인별 prior
- posterior strength 40/80/160, pitcher 결합비 75/90/100%

54개 raw 공식과 domain·가중치가 2022와 late-2023에서 동일하게 양수여야
했다. 선택된 `domain_latest_k80_p75`, R_ANCHOR, 0.2는 두 원점의 모든 월에서
양수였지만 full-2024 R_ANCHOR가 `-14.6297`이었다. 월별로는 5~6월이
`-15.08/-17.64`, 8~9월이 `+7.68/+15.46`이었다.

late-2024만 본 뒤 8~9월 route를 추가하는 것은 사후 선택이므로 실행하지
않았다.

## v40 — four-origin reliability gate

v39의 후기 개선이 당해 시즌 표본 축적 때문인지 독립적으로 확인했다.

1. 2021 raw BSS로 scope×prior family별 상위 2개 공식만 선택
2. 2022와 late-2023에서 동일 `minimum pitcher season_n`·domain·weight 합의
3. 고정 recipe를 full-2024에서 평가

후보 임계값은 `0/25/75/150/300`이었으나 선택값은 **0**이었다. 즉 과거 세
원점은 표본수 제한을 지지하지 않았고, 결과는 v39와 정확히 같았다. 후기
2024 개선은 단순 posterior reliability 증가로 설명되지 않는다. 또한 v39에서
2024 family 결과를 이미 본 상태이므로 v40은 수치가 좋아도 pristine 승격
감사로 간주하지 않도록 코드에서 패키징을 차단했다.

## v41 — joint pitch-type × failure-mode student

훈련행에서만 다음 ASOF snapshot으로 현재 투구의 구종 3종과 실패유형 4종을
복원해 12-class 공동 라벨을 만들었다. 추론 학생은 공식 ASOF·경기상황·손·팀
피처만 보고 공동 확률을 예측한다. 현재 투구 구종, 실패유형, TrackMan 값은
평가 입력에 없다.

- 공동 라벨 coverage: `99.8468%`
- 구종 정확도: 2022/23/24 `56.24% / 54.89% / 55.28%`
- 실패유형 정확도: `37.70% / 37.38% / 39.41%`
- 공동 정확도: `21.70% / 21.00% / 22.15%`
- 2022·late-2023 exact-recipe 합의 gate: **0개**
- fallback full-2024: `+0.1187`, 양수 월 비율 50%, 최악 월 `-0.2713`

첫 구현은 전체 시즌-state 재계산과 9회 12-class 학습으로 20분 상한을
초과했다. 라벨 결과를 보기 전에 공식 ASOF 원값+저비용 행 상태, 반감기 2의
단일 120-tree 학생으로 고정하고 연도별 체크포인트를 추가했다. 최종 재현
실행은 약 376초였고 원점별 중단 복구가 가능하다. 연구 모델과 체크포인트는
`artifacts/`에만 남긴다.

## 누수·선택 편향 감사

- 현재 행의 공식 ASOF snapshot만 사용하며 다른 평가 행을 조회하지 않는다.
- ASOF 누적 차감용 baseline은 audit season 이전 labelled train에서만 만든다.
- 현재 투구 구종·실패유형은 이전 시즌 학습 라벨로만 사용한다.
- v39/v41의 공식·route·weight는 2022와 late-2023에서 고정한 뒤 2024를 연다.
- v40은 2021을 공식 사전선별에 추가했지만 v39 family의 2024 결과 재사용
  위험을 별도 표시하고 패키징을 강제로 금지한다.
- Public 점수로 가중치나 월 route를 조절하지 않았다.

## 재시도 금지

새 독립 연도나 독립 팀 예측이 없으면 다음을 반복하지 않는다.

1. 직전 시즌·경력 prior를 이용한 current-season posterior의 k/선수비중 조정
2. v39를 8~9월 또는 late-season에만 사후 적용하는 route
3. pitcher current-season n 임계값의 추가 탐색
4. 단순 12-class 공동 구종×실패유형 학생의 tree/온도 미세조정
5. 2023·2024에서 선택된 v19 부모를 과거에 적용해 완전 OOF라고 부르는 것

## 다음 우선순위

저장소 내부의 ASOF lookup, 확률 보정, 현재 시즌 posterior, latent
pitch/failure, TrackMan student 신호는 v27 위에서 거의 포화됐다. 다음으로
필요한 것은 다음 중 하나다.

1. 동일 팀원이 독립적으로 만든 row-level OOF/test 예측과의 오차 공분산 감사
2. v19 route까지 각 원점 안에서 다시 선택하는 완전 nested base pipeline
3. 행 독립 규칙을 지키는 사전학습 표현 중 현재 모델과 생성 구조가 다른 모델

2026년 pitch-sequence world-model 연구는 순차 문맥의 잠재력을 보이지만,
이 대회에서는 다른 test 행을 연결할 수 없어 그대로 적용할 수 없다. 공동
잠재상태만 행별 학생으로 축약한 v41도 잔차 전이가 미미했다.

## 재현

```powershell
python -m src.archive.v39_hierarchical_season_forecast --project .
python -m src.archive.v40_reliability_gated_hierarchy --project .
python -m src.archive.v41_joint_pitch_failure_student --project .
python -m pytest -q
```

관련 연구:

- Ahn et al. (2026), *Neural Sabermetrics with World Model*:
  https://arxiv.org/abs/2602.07030
- Lee (2022), *Prediction of pitch type and location in baseball using an
  ensemble model of deep neural networks*:
  https://doi.org/10.3233/JSA-200559
- Doo & Kim (2018), hierarchical Bayesian matchup/log5:
  https://doi.org/10.1371/journal.pone.0204874
