# 투구 제구 성공 확률 예측 AI 온라인 해커톤

LG Aimers 9기 [**투구 제구 성공 확률 예측 AI 온라인 해커톤**](https://dacon.io/competitions/official/236743/overview/description) 5인 팀 솔루션.
대회 종료 후 최종 제출물과 연구 기록을 정리한 아카이브다.

## 최종 결과

> **리더보드 순위 20위 / 1,090팀** (상위 1.8%) · **Public 1182.94969702 (v345)**
> 첫 제출(749.5249) 대비 **+433.42**, exact-ASOF 기준선(1068.4366) 대비 **+114.51** 상승.

| 단계 | Public | 돌파 내용 |
|---|---:|---|
| 첫 제출 | 749.5249 | RandomForest 베이스라인 계열 |
| exact-ASOF 기준선 | 1068.4366 | v13, 시즌 간 전이되는 ASOF 분리 |
| 압력 EB 잔차 | 1093.3213 | v17, 투수 × 타자손 × count 압력 |
| 다년 state + 실패유형 routing | 1144.1518 | v19, 3-도메인 라우팅 확립 |
| 저분산 도메인 calibration | 1153.0436 | v22, row-local ASOF prior |
| shared pairwise FM | 1161.2021 | v84, 메커니즘 직교성 확보 |
| Public-quadratic stack | 1164.2949 | v124, 다축 스태킹 |
| conservative bridge | 1170.3015 | v148, 강한 부모 보존 전략 |
| mechanism-aware fallback | 1175.9747 | v244 |
| R_ANCHOR low-rank complement | 1181.7100 | v335, 목표 1180 돌파 |
| **선수 전이 + workload-H1 + Beta** | **1182.9497** | **v345, 최종 champion** |

본인 기여: EDA·특징 설계·검증·학습·앙상블·최종 재현 패키지의 전반을 담당했습니다. 순위와 점수는 5인 팀 성과이며, 개인 단독 성과로 표기하지 않습니다.

평가 지표: **Brier Skill Score 계열** — 확률 보정 품질을 재며, 값이 클수록 좋다.

```text
Brier Score = mean((예측 확률 - 실제 정답)^2)
BSS = max(0, 100000 × (1 - 모델 Brier Score / 기준 Brier Score))
```

최종 코드·검증 기록: `final_submission/` (**공개본에는 모델·고정 입력·추론 ZIP 미포함**). 기존 제출 추론 ZIP SHA-256
`D44578DC50220CE84DD4B8489BBAE680AFDCF93F931ED4236287B5A9E6F5AAAA`.

---

점수 1182.9497은 상수 평균 확률 기준보다 Brier 오차가 약 **1.183% 감소**했다는 뜻이며 정확도 1182%가 아닙니다. 리더보드 결과와 코드 검증·본선 진출은 구분합니다.

## 문제 정의

- **입력**: 한 행이 한 번의 투구 상황. 투수·타자 익명 ID, 경기 맥락(이닝·count·주자), 시즌 누적 ASOF 통계.
- **출력**: 해당 투구의 `P(control_success = 1)`.
- **데이터**: `train.csv` (2019~2024), `trackman_history.csv` (과거 투구 물리량), `test.csv`.
- **핵심 성질**: 정답 클래스를 자르는 문제가 아니라 **확률을 보정하는 문제**다. 극단적으로
  자신 있는 오답의 손실이 크므로, 최신 시즌에 과적합한 단일 고용량 모델보다 **검증된 부모를
  보존하면서 좁은 영역에만 작은 보정을 더하는** 전략이 유리했다.
- **결정적 제약**: 2022→2023에 큰 도메인 변화가 있다. 퓨처스리그(`F`)의 reverse 실패율이
  `4.82%` → `29.96%`로 급증해, 단일 전역 모델은 시즌 전이에서 무너진다.

---

## 솔루션 아키텍처

### 3-도메인 라우팅

전역 모델 대신 행을 세 도메인으로 나눠 각각 다른 보정을 적용한다. 도메인 경계는 공식
데이터의 `game_type`과 팀 ID만으로 결정되며 test 집계를 쓰지 않는다.

```text
F        = game_type == "F"                                    # 퓨처스리그(2군)
R_ANCHOR = game_type == "R" and (pitcher_team_id == 13 or batter_team_id == 13)
R_CORE   = 나머지 정규시즌 행
```

### 부모 보존 + 국소 보정

점수를 올린 모든 단계는 **부모를 수치적으로 보존한 채 한 도메인에만 작은 신호를 섞는**
형태였다. v345의 최종 수식은 부모 v335를 두 도메인에서 그대로 두고 `R_CORE`에만 새 축을 더한다.

```text
output[F]        = v335[F]                      # 보존
output[R_ANCHOR] = v335[R_ANCHOR]               # 보존
output[R_CORE]   = v335 기반 + 선수 전이 + 3시드 workload-H1 + 고정 Beta 셀
```

### 핵심 인사이트

- **로컬 이득은 Public 이득을 보장하지 않는다.** 2023 선택 gain이 `+505.7564`였던 후보가
  동결한 2024에서 `-2.8009`로 반전했다. 그래서 모든 후보에 다중 시간축·월/도메인·군집
  bootstrap 게이트를 걸고, **전이율**(로컬 이득 대비 실제 Public 이득)을 함께 봤다.
- **강한 부모를 건드리지 않는 것이 최선의 방어였다.** v148의 conservative bridge처럼
  보수적으로 되돌린 선택이 오히려 공격적 확장보다 높은 점수를 냈다.
- **직교성이 용량보다 중요했다.** v84의 shared pairwise FM처럼 기존과 다른 메커니즘을
  도입한 순간에 점수가 움직였고, 같은 계열의 강도 조정은 금방 한계에 부딪혔다.

### 시도했지만 효과 없음

| 시도 | 결과 |
|---|---|
| 잔차·다년 직접모형·latent state/failure mode·임베딩 신경망·그룹 강건 학습 (v23 구조 감사) | 여섯 계열 모두 외부 연도 게이트에서 기각 |
| 의미 기반 신규 신호와 Public 역산 가중치 (v24 의미 감사) | 시간축 게이트 통과 실패, `submit_v24`는 만들지 않음 |
| 손잡이 routing, 같은 계열 강도 미세 조정 (v26·v28) | 부모를 넘지 못해 기각. 이후 같은 계열 조정 중단 |
| v346~v378 후속 스크린 | v345를 넘지 못함. 기록은 `src/archive/`에 보존 |

---

## 폴더 구조

```text
final_submission/   v345 재현 코드·검증 기록 (비공개 자산 제외)
  code/             학습 코드 (고정 입력 별도 필요)
  inference/        추론 설명 (제출 ZIP 별도 필요)
  verification/     구성요소별 검증 결과
src/
  champion/         실제 제출된 버전으로 이어지는 계보
  core/             축·계약·패키징 공통 코드
  team_assets/      팀 자산 설명·학습 코드 (모델 제외)
  archive/          종료된 실험 모듈
tests/
  champion/ core/ archive/   src 구조를 그대로 반영
research/
  reports/          실험 결과와 제출 원장
  configs/          동결된 실험 및 패키징 계약
docs/               프로젝트 현황·인계·규칙
notebooks/          탐색 워크벤치와 챔피언 재현 노트북
scripts/            저장소 보조 감사
submissions/        제출 계보 문서
data/               공식 원본 데이터, Git 제외
artifacts/          모델·OOF 산출물, Git 제외 (manifest만 추적)
```

읽는 순서는 이 README → [`final_submission/README.md`](final_submission/README.md) →
[`src/champion/README.md`](src/champion/README.md)를 권한다. 앞의 둘이 최종 결과물이고,
세 번째가 거기에 이르는 코드 계보다.

---

## 재현 방법

### 0. 환경 설치 (Python 3.11)

```powershell
git clone https://github.com/jinwon25/Dacon.git
cd "Dacon/투구 제구 성공 확률 예측 AI 온라인 해커톤"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

### 1. 데이터 없이 되는 검증 (권장, 수십 초)

```powershell
python -m pytest -q                      # 데이터·자산 의존 테스트는 skip
python scripts/audit_repository.py       # 저장소 정책 감사
```

산출물이 없는 checkout에서는 해당 테스트가 자동으로 skip되므로, clone 직후 바로 돌아간다.

### 2. 공식 데이터 배치

원본은 저장소에 포함하지 않는다. DACON 대회 페이지에서 직접 내려받아 `data/`에 둔다.
파일 크기와 SHA-256은 [`data/README.md`](data/README.md)에서 대조한다.

```text
data/train.csv  data/trackman_history.csv  data/test.csv  data/sample_submission.csv
```

### 3. v345 구조 검사와 조건부 재현

```powershell
python -m pytest -q tests/champion/test_v345_build_transition_workload_beta_package.py
```

위 테스트는 구조·계약 검사이며 최종 점수 재현이 아닙니다. 상세 학습·체크포인트 재조립 절차는 [`final_submission/03_REPRODUCE.md`](final_submission/03_REPRODUCE.md)를
따른다. 고정 학습 체크포인트(약 297MB)는 공개하지 않습니다. 공식 데이터만 내려받아도 원 제출의 동일 재조립에 필요한 모든 입력이 갖춰지는 것은 아닙니다. CPU XGBoost 신규 학습은 원 CUDA 모델과 차이가 있으며, [구성요소별 검증](final_submission/verification/README.md)을 확인해야 합니다.

---

## 대회 규정 검토

대회와 코드 검증은 종료됐습니다. 코드 공개와 데이터 재배포를 구분해 검토한 [공개 판단·근거](../docs/PUBLICATION_REVIEW.md)를 확인하세요.

최종 추론은 **평가 행 하나의 투구 직전 정보와 공식 학습 데이터에서 동결한 artifact**를 사용하는 구조다. 아래 항목은 코드 구조와 기존 감사 기록의 범위이며 숨겨진 평가 실행을 새로 인증한 것은 아니다.

- 평가 행 간 독립: 행 순서·배치 크기를 바꿔도 같은 예측 (배치 불변성 최대 오차 `1.11e-16`)
- test 전체의 평균·순위·빈도·분포·순서·그룹 통계 미사용
- 외부 데이터 미사용, 추론 시 알 수 없는 oracle 정보 미사용
- 평가 시즌보다 미래인 target을 특징·보정·가중치 선택에 미사용
- 후보 선택·블렌드에는 Public 점수를 사용한 기록이 있음. 이를 독립 검증 점수로 해석하지 않으며, 과도한 probing 금지와 행별 추론 조건을 별도로 검토함

상세 감사는 [`research/reports/dacon_official_compliance_audit_20260816.md`](research/reports/dacon_official_compliance_audit_20260816.md),
승격 게이트 정의는 [`research/configs/evaluation_v3.json`](research/configs/evaluation_v3.json)에 있다.

---

## 데이터·산출물 정책

- 원본 DACON 데이터, 인증정보, 개인 키는 Git에 올리지 않는다.
- 대회 데이터에서 파생된 산출물(모델 ZIP·OOF·lookup·선수 연결표)도 **예외 없이** 제외한다.
  OOF의 정답·선수 ID와 제출 ZIP의 선수별 집계 prior를 공개하지 않기 위한 보수적인 정책이다. 모든 학습 모델의 공개가 규정상 일률 금지라고 단정하지 않는다.
- 저장소에는 각 산출물의 manifest만 남겨 무엇이 어디에 있는지 추적한다. 전달 규칙은
  [`docs/ARTIFACT_HANDOFF.md`](docs/ARTIFACT_HANDOFF.md)에 있다.
- 현재 프로젝트 정책은 `scripts/audit_repository.py`와 `tests/test_repository_audit.py`에서 검사합니다. 전체 과거 이력·JSON 모델·CSV 내용은 루트 `scripts/audit_public_repository.py --history`로 추가 검사합니다.

## 주요 문서

| 문서 | 용도 |
|---|---|
| [`final_submission/README.md`](final_submission/README.md) | 최종 제출 패키지 구성과 검증 범위 |
| [`final_submission/03_REPRODUCE.md`](final_submission/03_REPRODUCE.md) | v345 재현 절차 |
| [`src/champion/README.md`](src/champion/README.md) | 제출된 버전으로 이어지는 코드 계보 |
| [`research/reports/v345_public_result_20260831.md`](research/reports/v345_public_result_20260831.md) | 최종 champion 승격 근거 |
| [`research/reports/submissions.csv`](research/reports/submissions.csv) | 제출 ID·점수·해시 원장 |
| [공식 데이터 페이지](https://dacon.io/competitions/official/236743/data) | 데이터·컬럼 설명 직접 취득 (공개본에서 제외) |
| [`docs/PROJECT_MEMORY.md`](docs/PROJECT_MEMORY.md) | 프로젝트 불변 조건과 연구 원칙 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 브랜치·리뷰·커밋 규칙 |
