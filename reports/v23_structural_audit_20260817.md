# v23 구조 모델 감사 — 2026-08-17

## 결론

v22 Public 결과와 독립적으로 설계한 여섯 계열을 시간 순서대로 선택·감사했다. 단순 선형 보정부터 구조 잔차, 다년 직접모형, latent state/failure mode, 임베딩 신경망, 그룹 강건 학습, 2023 이후 전용 스플라인까지 범위를 넓혔지만 **동결한 2024 외부 감사에서 기준을 통과한 후보는 없었다.**

따라서 `submit_v23.zip`은 만들지 않았고 추가 제출도 하지 않는다. 현재 champion은 `submit_v22.zip`, 공식 Public은 `1153.0436023798`이다.

| 후보 | 선택 구간 gain | 동결 2024 gain | 월 양수 비율 | 최소 적용 도메인 gain | 판정 |
|---|---:|---:|---:|---:|---|
| nested 구조 잔차 LightGBM | 내부 최소 +179.2994 | **-63.6043** | 25.0% | -262.3788 | 기각 |
| 다년 직접 확률모형 | -5.5636 | **-0.7449** | 50.0% | -1.5137 | 선택 단계부터 기각 |
| 3단계 latent state/mode | -0.1183 | **-0.0478** | 25.0% | -0.1692 | 선택 단계부터 기각 |
| Brier 임베딩 신경망 | +154.1668 | **-67.4164** | 12.5% | -320.3247 | 기각 |
| season-domain 균등 신경망 | +197.3479 | **-44.2816** | 25.0% | -296.8389 | 기각 |
| 2023 이후 전용 spline-logistic | +505.7564 | **-2.8009** | 50.0% | -42.9736 | 기각 |

## 감사 원칙

1. 후보·학습 강도·적용 도메인은 2023의 정직한 v22 analogue OOF에서만 선택했다.
2. 선택을 동결한 뒤 2024 전체를 한 번만 외부 감사했다.
3. 외부 감사 통과 기준은 전체 gain `>= +5`, 양수 월 비율 `>= 75%`, 모든 적용 도메인 양수, 최악 월 `> -10`이다.
4. 2024 결과를 본 뒤 유리한 월·도메인·혼합 강도만 다시 고르는 행위는 금지했다.
5. 실제 제출용 후보는 외부 감사 게이트와 패키지 테스트를 모두 통과해야 한다.

이 구조는 시간 분포 변화가 있는 환경에서 무작위 분할 성능이 실제 미래 성능을 과대평가할 수 있다는 연구 결과와 같은 방향이다. 참고: [Temporal Distribution Shift in Model Assessment](https://proceedings.mlr.press/v235/han24b.html), [Future Gradient Descent](https://proceedings.mlr.press/v180/ye22b/ye22b.pdf).

## 확인된 구조 변화

연도별 성공률 자체가 단순한 잡음 수준을 넘어 이동했다. 특히 `F` 도메인은 2022→2023에 정의·구성 변화로 의심되는 큰 절벽이 있다.

| 연도 | 전체 | R_CORE | R_ANCHOR | F |
|---|---:|---:|---:|---:|
| 2022 | 0.528920 | 0.51172 | 0.47103 | 0.70875 |
| 2023 | 0.499957 | 0.49296 | 0.54365 | 0.47290 |
| 2024 | 0.486105 | 0.47990 | 0.52886 | 0.45928 |

이 변화 때문에 2019~2022에서 학습한 표현도, 2023 내부에서 매우 강한 표현도 2024로 넘어가며 부호가 바뀌었다. 후반기 잔차와 신경망의 큰 선택 gain은 일반화 가능한 신호라기보다 해당 시즌의 생성 구조를 학습한 것으로 판단한다.

## 실험별 해석

### 1. nested 구조 잔차

선수 ID를 제외한 저자유도 `context_l3`가 2023 내부 두 시간축에서 최소 `+179.2994`를 기록했지만, 2024에서는 `-63.6043`, 최악 월 `-448.8097`로 붕괴했다. 월·도메인 일관성도 모두 실패했다.

### 2. 다년 직접 확률모형

2019~2022로 학습하고 2023 후반에서 선택한 `l2_context_h2`의 2.5% 혼합은 선택 구간부터 `-5.5636`이었다. 2024 손실은 `-0.7449`로 작았지만 양의 개선 증거가 없으므로 기각했다.

### 3. latent state/failure-mode 3단계 선택

2022에서 38개 state/mode 신호를 12개로 사전 축소하고 2023에서 신호와 강도를 선택했다. 최종 `mode_lgb_h0.5_pow1`의 0.25% 혼합도 2023 `-0.1183`, 2024 `-0.0478`로 개선하지 못했다. 2022 latent failure-mode OOF는 별도 생성해 이후 다년 감사의 기반 자료로 보존했다.

### 4. 임베딩 신경망과 그룹 강건 학습

투수·타자·팀·카운트 임베딩과 ASOF 수치 특징을 Brier loss로 학습했다. 일반 recency 학습은 2023 `+154.1668`에서 2024 `-67.4164`, season-domain 균등 위험 학습은 `+197.3479`에서 `-44.2816`으로 반전했다. 그룹별 prior를 명시적으로 다루는 발상 자체는 타당하지만, 현재 데이터에서는 시즌 전환을 견디지 못했다. 참고: [Group-Aware Priors for Bayesian Neural Networks](https://proceedings.mlr.press/v238/rudner24a.html).

### 5. 2023 이후 전용 spline-logistic

`F`의 구조 단절을 피하려고 2019~2022 레이블을 완전히 버리고, 2023년 3~7월만 학습해 8~10월에서 모델·강도·도메인을 골랐다. `logistic C=0.1`, 30% 혼합, `R_ANCHOR+F` 적용은 선택 구간에서 `+505.7564`였으나 2024에서 `-2.8009`였다. `R_ANCHOR`만 보면 `+12.8102`였지만 `F`가 `-42.9736`이었다. 이 결과를 본 뒤 `R_ANCHOR`만 택하는 것은 외부 감사 누수이므로 제출 후보로 전환하지 않는다.

## Public 점수로 역산한 미세 조정의 한계

v21→v22의 Public gain과 2024 로컬 보정 크기를 단순 이차 손실로 근사하면, v22 보정의 Public 최적 배율은 약 `0.65`이고 v21 대비 기대 가능한 최대 gain은 약 `+2.13`이다. 현재 v22보다 추가로 얻을 수 있는 값은 약 `+0.60`에 불과하다.

이는 **가정에 의존한 진단값**이지 독립 검증 결과가 아니다. 같은 Public 결과를 사용해 배율을 고르고 다시 제출하는 것은 리더보드 과적합이며, 설령 근사가 맞아도 확인 당시 Top 10 격차 `4.9158`을 해소하지 못한다. 따라서 기존 보수형 ZIP이나 v22 강도 조정본은 제출하지 않는다.

## 야구 모델링 관점의 다음 유효 경로

현재 병목은 모델 용량이 아니라 관측되지 않은 투구 의도와 시즌별 생성 규칙 변화다. 투구 결과만으로 제구를 판단하면 실제 목표 지점과 우연히 들어간 공을 구분하기 어렵다. 최근 연구도 투구 의도 위치를 잠재변수로 다루는 방향을 제안한다. 참고: [xCTRL: A Framework for Evaluating Pitchers' In-Game Pitch Control](https://arxiv.org/abs/2508.19184), [A Hierarchical Bayesian Model of Pitch Framing](https://arxiv.org/abs/1704.00823).

다음 연구는 아래 순서로 제한한다.

1. 2022·2023·2024의 완전한 champion analogue OOF를 동일 recipe로 생성한다.
2. 투수×카운트×타자 손잡이별 잠재 목표 영역을 과거 투구만으로 추정하는 EM/계층 베이지안 표현을 만든다.
3. 한 시즌에서만 좋아지는 후보가 아니라 2022→2023과 2023→2024 두 전환에서 같은 방향인 후보만 남긴다.
4. 팀원이 보유한 독립 모델 OOF가 있으면 예측 상관과 오류 공분산을 기준으로 혼합한다. 같은 계보의 강도 조절본은 제외한다.
5. 위 조건으로 2024에서 최소 `+5`가 재현될 때만 2025 재학습과 `submit_v23.zip` 패키징을 재개한다.

## 재현 명령과 산출물

```powershell
python -m src.v23_structural_residual_screen --project . --output-dir artifacts/v23_structural_residual_20260817_01
python -m src.v23_multiyear_direct_screen --project . --output-dir artifacts/v23_multiyear_direct_20260817_01
python -m src.v23_three_stage_state_mode_screen --project . --output-dir artifacts/v23_three_stage_state_mode_20260817_01
python -m src.v23_neural_embedding_screen --project . --output-dir artifacts/v23_neural_embedding_20260817_02
python -m src.v23_neural_group_robust_screen --project . --output-dir artifacts/v23_neural_group_robust_20260817_01
python -m src.v23_postbreak_gam_screen --project . --output-dir artifacts/v23_postbreak_gam_20260817_01
python -m pytest -q
```

각 실험의 원수치는 대응하는 `artifacts/.../summary.json`에 있으며, 전체 테스트 결과는 `97 passed`다.
