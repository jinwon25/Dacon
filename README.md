# 투구 제구 성공 확률 예측

이 저장소는 DACON `Aimers 9기: 투구 제구 성공 확률 예측 AI 온라인 해커톤`에 참가하는 3인 팀의 공동 연구 공간이다. 코드, 실험 설정, 테스트, 검증 보고서와 협업 절차를 관리한다.

원본 데이터와 모델 파일은 일반 Git 이력에 넣지 않는다. GitHub에는 재현 가능한 코드와 선별된 실험 기록만 보관하고, 모델·OOF·제출 ZIP은 동일한 공식 DACON 팀원만 접근할 수 있는 별도 private artifact로 관리한다.

## 현재 현황

| 항목 | 현재 상태 |
|---|---:|
| 목표 Public 점수 | **1150** |
| 현재 최고 Public 점수 | **1144.1518063753** |
| 1150까지 남은 차이 | **5.8481936247** |
| champion 제출 파일 | `submit_v19.zip` |
| DACON 제출 ID | `1533965` |
| 제출 ZIP SHA-256 | `B12070D8017AE6A78BFC056F392BACDA931F8ED7439AA9821880FC986CE962B2` |
| 확인 당시 순위 | **16위** |
| 로컬 전체 추론 | **58.051초 / peak RSS 1,423.086MB** |
| 현재 기준 브랜치 | `main` |
| GitHub 운영 | feature branch → 팀 리뷰 → squash merge |
| 공식 결과 확인 시각 | `2026-08-16 13:15:31 KST` |

API의 `isSubmitted=true`, `detail=Success`와 공식 리더보드 반영을 모두 확인했다. 순위는 다른 참가자의 제출에 따라 달라질 수 있으므로 위 값은 확인 당시 기록이다. 상세 근거는 [`reports/v19_public_result_20260816.md`](reports/v19_public_result_20260816.md)에 있다.

v19는 v17의 `1093.3213473808`에서 **+50.8304589945** 상승했다. 새 실험의 운영 기준선은 v19이며, v17은 직접 부모로서 `submissions/history/`에 보존한다.

### 2026-08-16 Public 결과와 결정

| 제출 | Public 점수 | v17 대비 | 판단 |
|---|---:|---:|---|
| `submit_v13_fixed.zip` | 1068.4365711741 | -24.8847762067 | exact-ASOF 역사 기준선 |
| `submit_v17.zip` | 1093.3213473808 | 기준 | v19 직접 부모 |
| `submit_v18.zip` | 1090.4672420401 | -2.8541053407 | 공격적 F 확장 기각 |
| `submit_v19.zip` | **1144.1518063753** | **+50.8304589945** | 현재 champion |

v19의 다중 시즌 row-local 상태 앙상블과 latent failure-mode 라우팅이 Public에서도 유효했다. 이후 후보는 v19를 고정 부모로 두고, 사전 고정된 rolling-origin 검증을 통과한 독립 신호만 비교한다.

## 폴더를 처음 열었을 때

루트에는 현재 champion만 두고, 과거 제출물은 별도 보관한다.

```text
프로젝트 루트/
├─ submit_v19.zip          현재 champion
├─ submissions/history/   과거 제출본과 v19 부모 v17
├─ reports/README.md       최신 결과 문서 안내
├─ artifacts/README.md     모델 산출물 보존 기준
├─ src/                    학습·패키징·검증 구현
└─ tests/                  자동 검증
```

먼저 [`submissions/README.md`](submissions/README.md)와 [`reports/README.md`](reports/README.md)를 읽으면 현재 상태와 전체 이력을 빠르게 파악할 수 있다.

### 1. champion 파일 확인

```powershell
Get-FileHash -Algorithm SHA256 .\submit_v19.zip
```

결과가 `B12070D8017AE6A78BFC056F392BACDA931F8ED7439AA9821880FC986CE962B2`인지 확인한다.

### 2. 데이터와 환경 준비

```text
data/train.csv
data/trackman_history.csv
data/test.csv
data/sample_submission.csv
```

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

원본 데이터는 각 팀원이 DACON에서 직접 받아 `data/`에 둔다. 제출 ZIP, 모델, OOF와 인증정보는 Git에 커밋하지 않는다.

### 3. 검증 또는 연구 시작

```powershell
python -m src.validate_state_mode_joint
jupyter lab notebooks/experiment_workbench.ipynb
```

검증기는 루트의 v19와 `submissions/history/submit_v17.zip`, `artifacts/state_mode_joint_final_20260816/manifest.json`을 기본 입력으로 사용한다. 새 연구는 v19를 덮어쓰지 않고 별도 후보명으로 생성한다.

## 무엇을 예측하는 대회인가

데이터의 한 행은 한 번의 투구 상황을 나타낸다. 모델은 그 투구가 제구에 성공할 확률 `P(control_success = 1)`을 0과 1 사이의 값으로 출력한다.

이 문제는 단순히 성공과 실패를 맞히는 분류 문제가 아니다. 예를 들어 실제 성공한 투구에 대해 `0.90`을 예측한 모델은 `0.55`를 예측한 모델보다 더 좋은 확률 예측을 한 것이다. 반대로 실패한 투구에 `0.90`을 줬다면 큰 손해를 본다. 따라서 정답을 맞히는 것뿐 아니라 확률의 크기가 실제 빈도와 잘 맞는지도 중요하다.

평가는 Brier Skill Score 계열 지표를 사용한다.

```text
Brier Score = 예측 확률과 실제 정답 차이의 제곱 평균
BSS = max(0, 100000 × (1 - 모델 Brier Score / 기준 Brier Score))
```

Brier Score는 낮을수록 좋고 대회 점수는 높을수록 좋다. 극단적으로 자신 있는 오답은 손해가 크므로, 과적합한 단일 모델보다 잘 보정된 저분산 앙상블이 유리하다.

공식 BSS는 0에서 잘리므로, 어려운 과거 fold에서 두 모델의 Brier 차이를 잃지 않기 위해 후보 비교에는 `unclipped BSS-equivalent`를 쓰고 공식 점수는 별도로 표시한다. 고정 incumbent 대비 시즌별 paired 변화, 반감기 1년 가중 변화, 월·팀 집중도, 투수·타자·투수×타자 교차 bootstrap, 500/2,000/5,000개 연속 투구 block bootstrap과 final-family Reality Check를 함께 사용한다. v13·v17·v18 Public 관측은 서로 독립인 표본이 아니므로 고정 local→Public 환산계수를 만들지 않는다. 최신 평가 체계는 [`reports/local_evaluation_v2_20260815.md`](reports/local_evaluation_v2_20260815.md), v17/v18 사전·사후 비교는 [`reports/v17_v18_public_result_20260816.md`](reports/v17_v18_public_result_20260816.md)에 있다. 아래 점수 시나리오 계산은 v14/v15 당시의 역사적 판단을 재현할 때만 사용한다.

```powershell
python -m src.local_scorecard `
  --candidate v14 artifacts/v14_multiseason_anchor_20260815_01/v14_combined_metrics.csv artifacts/v14_multiseason_anchor_20260815_01/v14_combined_bootstrap.csv `
  --candidate v15 artifacts/v14_multiseason_anchor_20260815_01/v15_combined_metrics.csv artifacts/v14_multiseason_anchor_20260815_01/v15_combined_bootstrap.csv `
  --incumbent-public 1068.4365711741 `
  --calibration-local-gain 67.6 `
  --calibration-public-gain 43.8500585794
```

## 데이터에서 보는 정보

현재 방법론은 크게 다음 정보를 사용한다.

| 정보 묶음 | 예시 | 의미 |
|---|---|---|
| 경기 상황 | 이닝, 초·말, 볼·스트라이크, 아웃 | 투수가 스트라이크를 던져야 하는 압박 정도를 나타낸다. |
| 주자와 점수 | 베이스 상태, 점수 차, 주자 수 | 승부 상황과 투구 선택의 공격성을 나타낸다. |
| 선수 정보 | 투수·타자 ID, 좌우 유형 | 선수별 제구력과 상대 유형 차이를 반영한다. |
| 팀과 경기 구분 | 투수·타자 팀, `R`·`F` | 정규시즌과 포스트시즌, 특정 팀 변화점을 분리한다. |
| 누적 ASOF 기록 | 투수 성공률, 최근 경기 비율, 구종 구성 | 해당 투구 직전까지 관측된 선수 상태를 나타낸다. |
| TrackMan 계열 정보 | 투구 특성과 선수 프로필 | 기존 경기 기록과 다른 센서 기반 신호를 제공한다. |

가장 중요한 원칙은 현재 행을 예측하는 시점에 알 수 있는 정보만 사용한다는 것이다.

## 현재 방법론 한눈에 보기

v17은 처음부터 모든 것을 다시 예측하는 단일 모델이 아니다. v11→v13→v14로 이어진 안정적인 부모 예측에 최신 시즌 전문가와 조건부 잔차를 필요한 영역에만 적용한다. 아래 1~4단계는 v13 기반 예측을 만들고, 5단계가 v17의 추가 개선이다.

```text
한 투구 행과 그 시점까지의 과거 정보
                │
                ▼
      v11 앙상블 기본 확률 p11
                │
                ▼
        경기 영역을 세 가지로 분리
       ┌────────┼─────────┐
       ▼        ▼         ▼
    R_CORE   R_ANCHOR      F
       │        │          │
 최신 시즌    v11 보호   최신 시즌
 전문가 결합  그대로 유지 LGB 75% 결합
       └────────┼─────────┘
                ▼
    v14 보수적 F 조정 + v17 R_CORE 압박 잔차
                │
                ▼
       0~1 범위로 잘라 최종 확률 출력
```

## 1단계: v11 기본 예측

v11은 여러 모델이 서로 다른 오차를 보완하도록 만든 앙상블이다. 핵심 구성은 다음과 같다.

1. 기본 LightGBM과 Random Forest를 각각 35%, 65%로 결합한다.
2. 전체 성공률의 시즌 변화에 맞춰 logit 보정을 적용한다.
3. 정규시즌 `R`에는 최근 시즌을 더 크게 보는 recency Random Forest를 사용한다.
4. 경기 유형에 따라 TrackMan 전문가의 비중을 다르게 적용한다.
5. 포스트시즌 `F`에는 범주형 상호작용에 강한 CatBoost를 75% 결합한다.
6. 과거 OOF에서 검증된 domain residual과 legacy CatBoost 다양성 보정을 추가한다.

LightGBM은 복잡한 비선형 상호작용을 잘 잡고, Random Forest는 비교적 안정적인 평균을 만든다. CatBoost는 투수·타자·팀처럼 값의 종류가 많은 범주형 변수에 강하다. 서로 다른 모델을 섞으면 한 모델의 과도한 확신을 다른 모델이 완화할 수 있다.

v11 자체가 이미 공개 점수 `1024.5865`를 기록한 강한 기준선이므로, v13은 v11의 모든 행을 무조건 바꾸지 않는다.

## 2단계: exact-ASOF 최신 시즌 전문가

`ASOF`는 현재 시점까지라는 뜻이다. 예를 들어 2025년 5월의 한 투구를 예측한다면 2025년 6월 이후 결과는 사용할 수 없다.

데이터의 누적 통계에는 이전 시즌 기록과 현재 시즌 누적 기록이 함께 들어 있다. exact-ASOF 로직은 시즌 시작 전까지의 누적 기록을 별도 bank로 만들고, 현재 행의 누적값과 정확히 분리해 현재 시즌에서 실제로 쌓인 증거를 복원한다.

```text
현재 행까지 누적 기록
- 시즌 시작 전 누적 기록
= 현재 시즌에서 해당 투구 직전까지 쌓인 기록
```

이렇게 만든 특징으로 두 종류의 전문가를 학습한다.

- exact LightGBM: 범주형 정보와 비선형 관계를 포함해 최신 시즌 패턴을 잡는다.
- exact Ridge: 강한 규제를 걸어 작은 변화만 안정적으로 반영한다.

LightGBM은 표현력이 높지만 변동성이 있고, Ridge는 단순하지만 안정적이다. 두 모델을 작은 가중치로 함께 쓰면 최신 변화에 반응하면서도 v11에서 너무 멀리 벗어나지 않는다.

## 3단계: 경기 영역별 라우팅

모든 행에 같은 보정을 적용하면 평균적으로는 좋아져도 특정 영역이 크게 무너질 수 있다. 이를 막기 위해 검증 데이터의 변화 양상이 다른 세 영역을 분리한다.

| 영역 | 정의 | 2024 검증 행 수 | 적용 전략 |
|---|---|---:|---|
| `R_CORE` | 정규시즌 `R` 중 anchor 팀과 무관한 행 | 178,729 | v11에 최신 시즌 전문가와 안정 보정을 작게 추가한다. |
| `R_ANCHOR` | 정규시즌 `R` 중 익명 team ID 13이 포함된 행 | 44,768 | 변화점이 불안정해 v11을 그대로 보호한다. |
| `F` | 포스트시즌·파이널 계열 | 30,010 | exact LightGBM 쪽으로 75% 이동한다. |

team ID 13은 과거 시즌 전이에서 다른 팀보다 변화가 불안정했다. 이 영역에 최신 모델을 일괄 적용하면 특정 전이에서 손실이 커졌기 때문에, 현재 v13에서는 공격적으로 수정하지 않는다.

## 4단계: v13 최종 결합식

`p11`을 v11의 확률, `pLGB`를 exact LightGBM 확률, `pRidge`를 exact Ridge 확률이라고 두면 다음과 같이 계산한다.

### R_CORE

```text
p13 = p11
    + 직전 시즌 core bias
    + 0.125 × (pLGB - p11)
    + 0.20  × (pRidge - p11)
    + 0.20  × stable conditional correction
```

`전문가 - p11` 형태를 쓰는 이유는 v11을 버리지 않고 전문가 방향으로 일부만 이동하기 위해서다. 예를 들어 `0.125 × (pLGB - p11)`은 v11에서 LightGBM 쪽으로 12.5%만 이동한다는 뜻이다.

stable conditional correction은 `R_CORE`에서 v11이 반복적으로 틀리는 조건을 Ridge로 학습한 잔차 보정이다. 투수, 타자 손 유형, 3볼·2스트라이크 같은 압박 상황과 그 상호작용을 사용한다. 절편을 제거해 전체 확률을 무작정 올리거나 내리지 않고 조건별 모양만 보정한다.

### R_ANCHOR

```text
p13 = p11
```

검증에서 확신이 부족한 영역은 변경하지 않는 보호 전략이다.

### F

```text
p13 = p11 + 0.75 × (pLGB - p11)
    = 0.25 × p11 + 0.75 × pLGB
```

포스트시즌은 정규시즌과 분포가 다르고 최근 regime의 영향이 컸다. 과거 전이 검증에서 exact LightGBM의 개선이 강하고 일관됐기 때문에 이 영역에만 높은 가중치를 사용한다.

마지막에는 모든 값을 `0~1` 범위로 제한한다.

## 5단계: v17 다중 시즌 압박 잔차

v17은 v14의 보수적 F 조정을 부모로 사용하고 `R_CORE`에만 경험적 베이즈 잔차를 더한다. 그룹 키는 `투수 ID × 타자 손잡이 × 압박 상태`이며, 압박 상태는 `3볼`, `2스트라이크`, `일반`으로 나눈다. 2022~2024 OOF 잔차를 동일 가중으로 모으고 표본이 적은 그룹은 전역 평균 쪽으로 강하게 축소한다.

```text
p17 = p14 + 1.5 × EB_residual(pitcher, batter_hand, pressure)
EB alpha = 3200, 적용 영역 = R_CORE only
```

최신 전이에서 일반·3볼·2스트라이크 구간의 기여가 모두 양수였고, 2024 강화 bootstrap 최소 p05는 `+1.3534`, 최소 개선 확률은 `96.40%`였다. Public 점수도 v13 대비 `+24.8848` 상승해 이 잔차 계층을 채택했다. 단, 이 결과가 같은 계열의 추가 확장을 모두 정당화하지는 않는다.

## 6단계: v19 다중 시즌 상태·실패 유형 결합

v19는 v17을 부모로 유지하면서 2019~2024 데이터로 학습한 row-local 상태 앙상블과 latent failure-mode 라우팅을 `R_CORE`, `R_ANCHOR`, `F`별로 다르게 적용한다. 테스트 전체의 평균·빈도·순서에는 의존하지 않고 각 행에서 사용할 수 있는 정보만으로 보정한다.

- 사전 `near_1150` 게이트 7개와 패키지 검증 게이트 11개를 모두 통과했다.
- 최신 2024 rolling BSS-equivalent gain은 v17 대비 `+46.395981`이었다.
- 245,789행 전체 추론은 58.051초, peak RSS는 1,423.086MB였다.
- Public에서는 v17 대비 `+50.8304589945` 상승했다.

최종 모델과 해시는 `artifacts/state_mode_joint_final_20260816/`, 상세 판단은 [`reports/target1150_joint_candidate_20260816.md`](reports/target1150_joint_candidate_20260816.md)에 있다.

## 이 방법이 점수를 올린 이유

현재 결과는 다음 여섯 원칙이 함께 작동한 것으로 해석한다.

1. 강한 기존 모델을 부모로 유지해 기본 성능을 보존했다.
2. exact-ASOF로 최신 시즌 정보만 분리해 시간 변화에 대응했다.
3. 서로 다른 성격의 LightGBM과 Ridge를 결합해 분산을 낮췄다.
4. 불안정한 영역은 보호하고 근거가 강한 영역만 크게 수정했다.
5. 투수별 제구 오차를 타자 손잡이와 압박 카운트로 나누되, 큰 alpha로 축소해 희소 그룹의 과적합을 막았다.
6. 여러 시즌의 상태와 실패 유형을 함께 학습하되 도메인별 보호 규칙과 모델 합의를 적용했다.

v13은 v11 대비 Public `+43.8501`, v17은 v13 대비 `+24.8848`, v19는 v17 대비 `+50.8305`를 추가로 얻었다. 로컬 점수는 후보의 방향과 위험을 거르는 도구로 사용하며, 한 번의 Public 전달률을 고정 환산식으로 사용하지 않는다.

## 데이터 누수를 막는 규칙

다음 규칙은 성능보다 우선한다.

- 평가할 시즌보다 미래인 target을 특징이나 보정값에 사용하지 않는다.
- 현재 행 뒤에 있는 평가 데이터의 값이나 분포를 사용하지 않는다.
- 테스트 전체 평균, 순위, 빈도와 그룹 통계로 개별 행을 보정하지 않는다.
- calibration과 blend weight는 평가 시즌보다 앞선 rolling-origin OOF에서만 정한다.
- Public 점수에 맞춰 가중치를 역으로 미세 조정하지 않는다.
- 추론은 각 행만으로 결정되는 row-local 구조를 유지한다.

이 원칙 때문에 테스트 행 순서를 바꾸거나 여러 배치로 나눠도 같은 행의 예측은 변하지 않는다.

## 검증 방법

단일 2024 holdout 점수만 보고 후보를 선택하지 않는다. 현재 승격 게이트는 다음과 같다.

1. `2021→2022`, `2022→2023`, `2023→2024` 시즌 전이에서 방향이 일관적인지 확인한다.
2. `R_CORE`, `R_ANCHOR`, `F`별 개선과 회귀를 따로 확인한다.
3. 월별 결과를 확인해 특정 월의 큰 손실을 걸러낸다.
4. bootstrap으로 개선 분포와 하위 5%를 확인한다.
5. 행 순서와 배치 크기를 바꿔도 예측이 같은지 확인한다.
6. 제출 ZIP의 계보, offline 실행, 시간과 메모리를 확인한다.

v13의 제출 전 검증 결과는 다음과 같다.

- 2024의 8개 월 구간이 모두 양수
- 전체 bootstrap 개선 하위 5% 약 `+47.9`
- 245,789행 추론 `32.193초`
- peak RSS `1,418.4MB`
- 분할 배치 최대 절대 차이 `1.110e-16`
- `R_ANCHOR`의 v11 예측 보존 확인
- 외부 네트워크와 테스트 배치 집계 미사용 확인

상세 결과는 [`reports/v13_fixed_recent_exact_validation.md`](reports/v13_fixed_recent_exact_validation.md)와 [`reports/v13_public_result_20260815.md`](reports/v13_public_result_20260815.md)에 있다.

## 현재 한계와 다음 연구 방향

1150까지는 `5.8482`점이 남아 있다. v19의 큰 상승에도 2024가 반복 사용된 개발 구간이라는 한계와 한 번의 Public 관측만 있다는 불확실성은 그대로다.

우선순위는 다음과 같다.

1. v19를 고정 incumbent로 두고 독립적인 신호만 nested rolling-origin에서 비교한다.
2. `R_CORE`, `R_ANCHOR`, `F`의 변경을 분리해 어느 영역의 개선인지 식별한다.
3. v18에서 실패한 공격적 F 가중치와 Public 점수 기반 미세 조정은 재사용하지 않는다.
4. 다중 시즌 전이, 월별 결과, 교차·block bootstrap과 final-family Reality Check를 모두 통과한 후보만 제출한다.

v12의 legacy curvature 보정과 단순 F/R expert routing은 공개 점수가 하락했으므로 다시 사용하지 않는다. `R_ANCHOR` 전체에 최신 모델을 일괄 적용하는 방식과 v18의 공격적 F 부모도 현재는 보류한다.

## 처음 참여할 때의 순서

1. GitHub 저장소 초대를 수락한다.
2. [`docs/GITHUB_START_GUIDE.md`](docs/GITHUB_START_GUIDE.md)를 읽고 저장소를 clone한다.
3. Python 3.11 환경과 개발 의존성을 설치한다.
4. DACON에서 원본 데이터를 직접 내려받아 `data/`에 넣는다.
5. [`submissions/README.md`](submissions/README.md)와 [`reports/README.md`](reports/README.md)에서 champion과 실험 계보를 확인한다.
6. `notebooks/experiment_workbench.ipynb`를 열고 기본 안전 설정으로 전체 셀을 실행한다.
7. 저장소 감사와 데이터 비의존 테스트를 통과시킨다.
8. 최신 `main`에서 새 작업 브랜치를 만든다.

```powershell
git clone https://github.com/Lg-Aimers-chungang/hackathon.git
cd pitch-control-probability
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install jupyterlab
```

원본 데이터는 다음 위치에 둔다.

```text
data/train.csv
data/trackman_history.csv
data/test.csv
data/sample_submission.csv
```

파일 크기와 SHA-256은 [`data/README.md`](data/README.md)를 기준으로 확인한다.

Python 스크립트의 공식 실행 진입점은 아래 노트북 하나다.

```powershell
jupyter lab notebooks/experiment_workbench.ipynb
```

처음에는 `DRY_RUN=True`, `RUN_HEAVY=False`, `RUN_PACKAGING=False`를 유지한다. 이 노트북이 경로 확인, 테스트, screen, 강건 평가, 학습, 패키징과 검증 스크립트를 순서대로 호출한다.

```powershell
python scripts/audit_repository.py --include-untracked
python -m pytest -q tests/test_metrics.py tests/test_local_scorecard.py tests/test_robust_local_evaluation.py tests/test_v16_residual.py tests/test_experiment_notebook.py tests/test_features.py tests/test_followup.py tests/test_hierarchical.py tests/test_recency_training.py tests/test_rolling_drift.py tests/test_residual.py tests/test_target_encoding_audit.py tests/test_top1100_features.py tests/test_trackman_linkage.py tests/test_trackman_soft_linkage.py tests/test_repository_audit.py
```

데이터까지 준비된 환경에서는 전체 테스트를 실행한다.

```powershell
python -m pytest -q
```

## 먼저 읽을 파일

| 경로 | 용도 |
|---|---|
| `README.md` | 문제, 현재 점수, 방법론과 시작 순서를 설명한다. |
| `docs/PROJECT_STATUS.md` | 최고점 모델, 검증 결과와 다음 연구 방향을 기록한다. |
| `docs/EXPERIMENT_WORKFLOW.md` | 팀 공통 실험·검증·패키징 순서를 설명한다. |
| `docs/GITHUB_START_GUIDE.md` | GitHub를 처음 사용하는 팀원의 작업 순서를 설명한다. |
| `CONTRIBUTING.md` | 브랜치, Pull Request와 실험 기록 규칙을 설명한다. |
| `notebooks/experiment_workbench.ipynb` | 현재 실험 스크립트를 안전하게 호출하는 실행 워크벤치다. |
| `src/recent_shared_exact_asof.py` | exact-ASOF 시계열 검증을 구현한다. |
| `src/train_recent_exact_overlay.py` | v13 overlay 모델을 학습한다. |
| `src/train_v16_residual.py` | v17의 다중 시즌 경험적 베이즈 잔차를 학습한다. |
| `src/evaluate_v16_robust.py` | 잔차 후보의 다중 의존성·선택편향 검증을 수행한다. |
| `src/package_v16_residual.py` | 검증된 잔차 artifact를 부모 ZIP에 결합한다. |
| `src/validate_v16_residual.py` | v17/v18 계열 ZIP의 계보·실행·불변성을 검증한다. |
| `src/v10_overlay_script.py` | v11과 v13의 실제 추론 파이프라인이다. |
| `src/package_recent_exact_overlay.py` | v11 부모와 v13 artifact를 결합한다. |
| `src/validate_recent_exact_overlay.py` | 제출 ZIP의 실행, 계보와 배치 불변성을 검증한다. |
| `src/package_state_mode_joint.py` | v17 부모와 최종 state/mode artifact를 결합해 v19를 만든다. |
| `src/validate_state_mode_joint.py` | v19 계보·실행·불변성과 자원 사용량을 검증한다. |
| `reports/submissions.csv` | 지금까지의 제출 이력을 기록한다. |

## 폴더 구조

```text
.github/                Pull Request 양식과 자동 CI 설정
configs/                실험별 설정 파일
data/                   각자 받은 DACON 원본 데이터, Git 제외
docs/                   시작 안내, 프로젝트 현황과 인계 문서
notebooks/              스크립트를 호출하는 팀 실험 워크벤치
reports/                실험 결과, 검증 보고서와 제출 기록
scripts/                저장소 안전성 등 보조 점검 도구
src/                    학습, 검증, 패키징과 추론 구현
submissions/history/    과거 제출본과 v19 부모 ZIP, Git 제외
tests/                  데이터 비의존 테스트와 로컬 통합 테스트
artifacts/              모델과 OOF 산출물, Git 제외
model/                  추론용 모델, Git 제외
output/                 생성된 예측 결과, Git 제외
```

과거 실험 코드는 비교와 실패 기록을 보존하기 위해 남겨 둔다. 새 작업은 [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md)의 활성 파일 목록을 먼저 확인한 뒤 시작한다.

## GitHub 협업 규칙

- `main`은 검증된 기준선으로 유지한다.
- `main`에 직접 push하지 않고 `exp/<가설>`, `fix/<수정>`, `docs/<주제>` 브랜치를 사용한다.
- 한 Pull Request에는 하나의 가설이나 하나의 수정만 담는다.
- 병합 전에 최소 한 명의 팀원 리뷰와 CI 성공을 확인한다.
- 저장소는 squash merge만 허용하며 병합된 원격 브랜치는 자동 삭제된다.
- DACON 제출은 지정된 담당자 한 명이 진행한다.
- 제출 결과는 `reports/submissions.csv`에 기록한다.

현재 요금제에서는 private 저장소의 branch protection을 강제할 수 없다. 따라서 `main` 직접 push 금지는 팀 규칙으로 지켜야 한다.

## Git 이력에 넣지 않는 파일

다음 파일은 private 저장소라도 일반 Git commit에 포함하지 않는다.

```text
data/ 원본 파일
model/ 모델 파일
artifacts/ OOF와 학습 산출물
output/ 예측 결과
submit*.zip
submissions/**/*.zip
.env 및 API token
개인 쿠키, 인증서와 key 파일
```

동일한 공식 DACON 팀으로 등록된 팀원끼리 모델과 OOF를 공유할 때는 Git 이력 대신 [팀 전용 Google Drive]([private reference removed])를 사용한다. Drive의 일반 액세스는 반드시 `제한됨`으로 두고, 공식 팀원 계정만 개별 초대한다. 원본 DACON 데이터와 인증정보는 Drive에도 올리지 않는다. v17 인계 자료는 v19의 부모 계보 재현용으로 보존하며, 현재 로컬 파일 구조는 [`submissions/README.md`](submissions/README.md)를 기준으로 한다.

커밋 전에는 항상 다음 명령을 실행한다.

```powershell
git status
python scripts/audit_repository.py --include-untracked
```

## 추가 문서

- [`docs/GITHUB_START_GUIDE.md`](docs/GITHUB_START_GUIDE.md): GitHub 용어와 일일 작업 순서
- [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md): 모델 현황과 다음 연구 우선순위
- [`docs/REFERENCE_MAP.md`](docs/REFERENCE_MAP.md): 현재 수치, 대회 규칙과 연구 참고자료의 출처 지도
- [`docs/EXPERIMENT_WORKFLOW.md`](docs/EXPERIMENT_WORKFLOW.md): 팀 공통 실험 실행·검증 절차
- [`docs/ARTIFACT_HANDOFF.md`](docs/ARTIFACT_HANDOFF.md): v19 부모인 v17과 v13 과거 기준선의 Drive 구조·해시·인계 방법
- [`docs/TEAM_ONBOARDING.md`](docs/TEAM_ONBOARDING.md): 팀원 초대 후 확인할 체크리스트
- [`CONTRIBUTING.md`](CONTRIBUTING.md): 실험과 Pull Request 작성 규칙

## 2026-08-16 인계 요약

- Public champion: `submit_v19.zip`, `1144.1518063753`
- 확인 당시 순위: 16위, 제출 ID `1533965`
- 목표 1150까지: `5.8481936247`
- 채택: 다중 시즌 row-local 상태 앙상블 + latent failure-mode 도메인 라우팅
- 운영 기준: v19 고정 비교와 구성 요소 단위 검증
- Public 근거: [`reports/v19_public_result_20260816.md`](reports/v19_public_result_20260816.md)
