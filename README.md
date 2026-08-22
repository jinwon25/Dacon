# 투구 제구 성공 확률 예측

이 저장소는 DACON `Aimers 9기: 투구 제구 성공 확률 예측 AI 온라인 해커톤`에 참가하는 3인 팀의 공동 연구 공간이다. 코드, 실험 설정, 테스트, 검증 보고서와 협업 절차를 관리한다.

원본 데이터와 인증정보는 GitHub에 올리지 않는다. 현재 비공개 저장소에는 공식 DACON
팀원 3인만 접근하며, 확정 1162 모델 ZIP과 fidelity-labelled OOF evidence만 Git LFS로
선별 보관한다. 나머지 모델·OOF·실험 캐시는 일반 Git 이력과 LFS 모두에서 제외한다.

> **2026-08-22 최신 기준**: v84 위 R_CORE 안정 구간에 독립 FM과 paired conditional
> correction을 적용한 v104가 Public **1162.6302840289**를 기록했다. 현재 전달·실행
> 기준은 과거 ZIP 계보가 아니라
> `artifacts/standalone_champion_1162/standalone_champion_1162.zip`이다. 아래
> 2026-08-18 정정은 v26 이전 계보를 설명하는 역사 기록이며 최신 기준을 대체하지
> 않는다. 최신 후속 실행 결과와 1170 로드맵은
> [`reports/target1170_v104_public_result_20260822.md`](reports/target1170_v104_public_result_20260822.md)를 따른다.

> 다음 작업 세션의 불변 조건은
> [`docs/PROJECT_MEMORY.md`](docs/PROJECT_MEMORY.md)에 영구 기록한다.
> 최종 gate OOF, v62~v64 직교·물리 후보, 공개 대회 자료와 최신 논문 감사 결과는
> [`reports/target1170_research_update_20260822.md`](reports/target1170_research_update_20260822.md)를
> 참고하고, 후보 승격 기준은
> [`configs/evaluation_v3.json`](configs/evaluation_v3.json)을 최신 원본으로 사용한다.

> **2026-08-18 champion 정정**: 아래 표와 이어지는 서술은 이전에 `submit_v27.zip`을 champion으로 기록했지만, 공식 DACON 제출 이력을 다시 대조한 결과 이는 오류였다. `1535195`는 DACON이 실제 발급하는 5자리 제출 ID 형식과 다르다. **실제 champion은 `submit_v26.zip`**이다 — Public **1157.9736407889**는 그대로지만, 제출 ID는 `51773`(제출 2026-08-17 05:08:58)이다. `submit_v27.zip`은 Public `1156.6153781694`(ID `51775`), `submit_v28.zip`은 `1156.9983034655`(ID `51783`)로 둘 다 v26보다 낮다. 재현·검증은 [`notebooks/v26_champion_reproduction.ipynb`](notebooks/v26_champion_reproduction.ipynb)를 기준으로 한다. 정정 근거는 [`reports/target1170_followup_20260817.md`](reports/target1170_followup_20260817.md) 상단과 [`reports/submissions.csv`](reports/submissions.csv)에 있다.

## 현재 현황

| 항목 | 현재 상태 |
|---|---:|
| 목표 Public 점수 | **1170** |
| 현재 최고 Public 점수 | **1162.6302840289** |
| 1170까지 남은 차이 | **7.3697159711** |
| 1200까지 남은 차이 | **37.3697159711** |
| 확인 당시 10위 컷 / 격차 | 미확인 (정정 이전 값은 v27 기준 기록이므로 재확인 필요) |
| champion 전달 파일 | `artifacts/standalone_champion_1162/standalone_champion_1162.zip` |
| 모델 계보 | `v25 → v26(.15) → TrackMan-ASOF → EXP-021 R_CORE(.10) → shared FM F(.10) → v104 stable R_CORE` |
| DACON 제출 ID | `60626` |
| 단일 ZIP SHA-256 | `0C3B6A9D88D31D9AC32FB43FFD642FA07BFCC3367699FA71841E35188FBDA0BA` |
| private artifact | Git LFS 모델 ZIP + `artifacts/oof_champion_1161/` evidence |
| 확인 당시 순위 | **13위** |
| 단일 ZIP 검증 | 83개 파일·CRC·행 독립성 통과, 245,789행 proxy 80.485초, 공식 제출 48초 |
| 현재 기준 브랜치 | `team/main` (이 릴리스 커밋) |
| GitHub 운영 | feature branch → 팀 리뷰 → squash merge |
| 공식 결과 확인 시각 | `2026-08-22 23:06 KST` |

공식 리더보드에서 점수를 확인했다. 순위는 다른 참가자의 제출에
따라 달라질 수 있다. 상세 근거는
[`reports/target1170_v104_public_result_20260822.md`](reports/target1170_v104_public_result_20260822.md)에 있다.

### 1158.07 대비 무엇이 달라졌나

직전 팀 최고점 `1158.0745556751`의 예측을 폐기하거나 전면 재학습한 것이 아니다.
검증된 1158 계보를 부모로 고정하고, 서로 겹치지 않는 두 구간에 작은 logit 보정을
순차 적용했다.

| 단계 | 적용 행 | 추가 모델·강도 | Public | 직전 대비 |
|---|---|---|---:|---:|
| 1158 기준 | 기존 `R_ANCHOR` TrackMan-ASOF gate 포함 | 기존 계보 | 1158.0745556751 | - |
| v82 | `R_CORE`만 | 사전 동결 EXP-021 strict, logit `0.10` | 1159.3352239501 | +1.2606682750 |
| v84 | `F`만 | full-2023+late-2024, source·기간 균형 rank-16 shared pairwise FM, logit `0.10` | 1161.2020600422 | +1.8668360921 |
| v104 | 안정 `R_CORE`만 | 두 source FM 평균 + paired conditional correction, 각각 `0.10` | **1162.6302840289** | +1.4282239867 |

1158 대비 누적 상승은 **+4.5557283538**이다. `R_ANCHOR`는 그대로 보호했고,
v104는 count·pitcher history·platoon 중 두 조건 이상이 안정적인 R_CORE 행만 추가로
수정했다. 개선점은 전역 확률 이동이 아니라 검증된 하위 집단별 잔차 보정을 결합한 것이다.

### 최고점 단일 버전 재검증

`standalone_champion_1162.zip`은 루트에 `script.py`, `requirements.txt`, `model/`만
가진 83개 파일짜리 독립 패키지다. 과거 제출 ZIP이나 저장소 `src/`를 import하지 않는다.
실제 확정 ZIP을 임시 디렉터리에 풀어 그 안의 코드와 모델만으로 다시 검사한 결과는
다음과 같다.

- SHA-256: `C033FC38A5F9681E45B0BD2494359B8BD1387EE44E5F318C5B7A5547CFE6C4F7`
- ZIP CRC와 루트 구조: 통과
- 공식 규칙 정적 검사: 금지 연산 0건
- 5행 실데이터의 단일행·순서변경·분할 예측 최대 차이: 각각 `1.11e-16`, `0`, `0`
- 245,789행 규모 격리 재실행: `114.487초`, 유한값·`[0, 1]` 범위·ID 정렬 통과
- DACON 실제 제출 실행시간: `41초`로 공식 제한 `600초` 이내

동일 검사는 아래 한 명령으로 반복할 수 있다. `--scale-rows`를 생략하면 빠른 샘플·행
독립성 검사만 수행한다.

```powershell
python -m src.audit_standalone_release `
  --package artifacts/standalone_champion_1161/standalone_champion_1161.zip `
  --test-csv data/test.csv `
  --scale-rows 245789 `
  --timeout-seconds 180
```

원본 데이터와 인증정보는 대회 데이터 보호를 위해 GitHub에 올리지 않는다. 확정 ZIP과
선별 OOF evidence는 이 비공개 저장소의 Git LFS로 공식 팀원에게만 전달하고, 일반 Git에는
해시·manifest·생성 코드·검증 코드와 결과 문서를 남긴다.

### 2026-08-22 평가 재감사

- 공식 BSS 구현은 맞다. 1170에는 테스트 양성률을 0.4861로 가정할 때 평균 Brier
  약 `2.1978e-05` 감소가 필요하다.
- 2024 label은 반복 실험에 사용돼 독립 holdout이 아니다. 이후에는
  `development_contaminated` 진단으로만 사용한다.
- 새 후보는 서로 다른 nested/locked primary 축 2개, 월·domain·세 종류 bootstrap,
  final-family Reality Check를 모두 통과해야 한다.
- local→Public 단일 환산은 금지하며 clean 관측 3개 전에는 projection을 출력하지 않는다.
- 공식 추론 제한은 600초이고 저장소의 120초는 내부 soft guard다.
- v75에서 전역 calibration은 이미 양호했고, v76 domain×count source-only 보정은
  exact late23→full24에서 `-45.486`으로 기각됐다. 해당 계열은 재개하지 않는다.
- v77 팀 exact OOF 계약과 constrained blend evaluator를 완성했다. 독립 OOF가 없어 실제
  weight는 계산하지 않았다.
- v78 환경 안정 잔차는 primary 두 축에서 eta `0`, 미래 감사축에서 `-7.5362`로 기각했다.
  제출 ZIP과 Public probe는 만들지 않았다.
- v82는 full-2024 OOF gain `+1.2596972`와 실제 Public gain `+1.2606643`이 거의 정확히
  일치해 새 champion으로 승격했다. 이 한 건을 다른 family의 환산계수로 쓰지 않는다.
- v83 새 부모 조건부 감사에서 strict 증량과 v50은 기각했고, F-only v56을 다음 고정
  recipe 복원 1순위로 정했다.
- v84는 그 고정 v56 레시피를 full-2023+late-2024 source로 복원해 단일 ZIP으로 제출했고,
  Public `+1.8668360921`이 전이돼 **1161.2020600422** 새 champion으로 승격했다.

최종 TrackMan-ASOF gate는 v26에서 **+0.1009148862** 상승했다. 현재 후속
실험의 운영 기준선은 v84 Public 1161.2020600422이며, v82·v26·1158 standalone과 v25는 계보
감사용 부모로만 보존한다. v29~v56 중 상당수는 eta `0.10` 부모를 사용했으므로 현재
기준선 위 후보로 간주하지 않는다. v57 strict 하나만 최종 부모 위에 재계산해 v82가 됐다.

v26은 v25의 `1155.8293405409`에서 **+2.1443002480**, v22에서 **+4.9300384091** 상승했다(위 정정 참고: 이 gain은 원래 v27 몫으로 잘못 기록됐던 값과 같은 크기이며, 실제로는 v26의 것이다). 후속 v27·v28은 검증 후 실제 제출했지만 v26을 넘지 못했다.

2026-08-17 후속 연구에서 row-local 도메인 보정과 투수·타자 ASOF prior를 결합한 `submit_v22.zip`을 생성하고 13개 패키지 게이트를 모두 통과시켰다. Public에서 v21 대비 `+1.5298`이 전이돼 새 champion으로 승격했다. 사전 등록한 `0 ~ +3` 분기에 따라 같은 계열의 강도 조정과 보수형 제출은 중단한다. 상세 판단은 [`reports/v22_public_result_20260817.md`](reports/v22_public_result_20260817.md)에 있다.

후속으로 잔차·다년 직접모형·latent state/failure mode·임베딩 신경망·그룹 강건 학습·2023 이후 전용 spline-logistic까지 여섯 구조 계열을 감사했다. 2023 선택 gain이 최대 `+505.7564`였던 후보조차 동결한 2024에서 `-2.8009`로 반전했고, 나머지도 모두 외부 게이트를 통과하지 못했다. `submit_v23.zip`은 만들지 않았다. 핵심 병목은 모델 용량보다 2022→2023의 큰 도메인 변화와 시즌 전이 실패다. 상세 결과는 [`reports/v23_structural_audit_20260817.md`](reports/v23_structural_audit_20260817.md)에 있다.

공식 FAQ와 데이터 의미를 다시 감사해 `F=Futures(퓨처스리그/2군)`, `R=Regular(1군 정규시즌)`로 문서를 교정했다. ASOF 누적은 전 행에서 직전 통계와 정확히 일치했고, F target 급락은 reverse 실패율의 2022 `4.82%`→2023 `29.96%` 급증과 함께 나타났다. 허용된 TrackMan ID 연결은 이미 train의 `82.54%`를 정렬하고 있었으며, 새 의미 기반 신호와 Public 역산 가중치는 시간축 게이트에서 기각됐다. `submit_v24.zip`은 만들지 않았다. 상세 결과는 [`reports/v24_semantic_eda_20260817.md`](reports/v24_semantic_eda_20260817.md)에 있다.

### 2026-08-22까지의 Public 결과와 결정

| 제출 | Public 점수 | v17 대비 | 판단 |
|---|---:|---:|---|
| `submit_v13_fixed.zip` | 1068.4365711741 | -24.8847762067 | exact-ASOF 역사 기준선 |
| `submit_v17.zip` | 1093.3213473808 | 기준 | v19 직접 부모 |
| `submit_v18.zip` | 1090.4672420401 | -2.8541053407 | 공격적 F 확장 기각 |
| `submit_v19.zip` | 1144.1518063753 | +50.8304589945 | v20 직접 부모 |
| `submit_v20.zip` | 1151.472428719 | +58.1510813382 | v21 직접 부모 |
| `submit_v21.zip` | 1151.5138356157 | +58.1924882349 | v22 직접 부모 |
| `submit_v22.zip` | **1153.0436023798** | **+59.7222549990** | v25 직접 부모 |
| `submit_v25.zip` | **1155.8293405409** | **+62.5079923062** | v26 직접 부모 |
| `submit_v26.zip` | **1157.9736407889** | **+64.6522934081** | 최종 gate의 직접 부모 (ID `51773`) |
| `submit_v27.zip` | 1156.6153781694 | +63.2940307886 | v25 기반 10% probe, v26보다 낮아 기각 |
| `submit_v28.zip` | 1156.9983034655 | +63.6769560847 | v27 기반 손잡이 routing, v26보다 낮아 기각 |
| `0819_3_tmgate03.zip` | 1158.0745556751 | +64.7532085624 | 직전 champion, 단일 ZIP 감사본 보존 |
| `submit_v82_probe.zip` | **1159.3352239501** | **+66.0138765693** | v84 직접 부모, `standalone_champion_1159.zip` |
| `submit_v84_probe.zip` | **1161.2020600422** | **+67.8807126614** | **현재 champion**, 로컬 전달명 `standalone_champion_1161.zip` |

v26은 v25의 post-break R_ANCHOR 직접확률 보정 강도를 15%로 적용해 Public
`+2.1443`을 추가했다. 이후 시도한 10% probe(v27)와 손잡이 routing(v28)은 모두
v26에 못 미쳤다. 최종 TrackMan-ASOF gate가 v26에서 `+0.1009`를 더해 당시
1158.0746 기준선을 만들었다.

v82는 이 기준선의 R_CORE에 사전 고정 EXP-021 strict를 10% 혼합해 Public
`+1.2606682750`를 추가했다. v84는 F-only shared FM으로 다시 `+1.8668360921`을 더했다.
이후 후보는 1161.2020600422를 정확한 새 부모로 사용한다.

## 폴더를 처음 열었을 때

현재 작업은 단일 champion과 그 manifest를 기준으로 한다. 과거 제출물은 계보
감사에만 사용하며 새 팀원에게 전달하지 않는다.

```text
프로젝트 루트/
├─ artifacts/standalone_champion_1161/
│  ├─ standalone_champion_1161.zip   현재 전달·실행 파일
│  ├─ standalone_manifest.json       파일 해시·검증 결과
├─ artifacts/standalone_champion_1159/  직전 챔피언 감사본
├─ artifacts/standalone_champion_1158/  직전 챔피언 감사본
├─ docs/STANDALONE_CHAMPION.md        실행·전달 절차
├─ reports/README.md                  최신 결과 문서 안내
├─ submissions/                       과거 제출 계보 안내
├─ src/                               학습·패키징·검증 구현
└─ tests/                             자동 검증
```

먼저 [`submissions/README.md`](submissions/README.md)와 [`reports/README.md`](reports/README.md)를 읽으면 현재 상태와 전체 이력을 빠르게 파악할 수 있다.

### 1. champion 파일 확인

```powershell
git lfs pull --include="artifacts/standalone_champion_1161/*,artifacts/oof_champion_1161/*"
Get-FileHash -Algorithm SHA256 `
  .\artifacts\standalone_champion_1161\standalone_champion_1161.zip
```

결과가 `C033FC38A5F9681E45B0BD2494359B8BD1387EE44E5F318C5B7A5547CFE6C4F7`인지 확인한다.

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

원본 데이터는 각 팀원이 DACON에서 직접 받아 `data/`에 둔다. 인증정보는 절대
커밋하지 않는다. 모델·OOF는 현재 1161 챔피언의 명시된 LFS allowlist만 허용한다.

### 3. 검증 또는 연구 시작

```powershell
python -m pytest -q tests/test_package_standalone_champion.py
```

가장 먼저 단일 릴리스의 구조·실행·행 독립성을 검증한다. v26 재현 노트북과
v27·v28 검증기는 역사 계보를 감사할 때만 사용한다. 새 연구는 현재 standalone
champion을 덮어쓰지 않고 별도 후보명으로 생성한다.

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
| 팀과 경기 구분 | 투수·타자 팀, `R`·`F` | 공식 의미인 1군 정규시즌과 퓨처스리그, 특정 팀 변화점을 분리한다. |
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
5. 퓨처스리그 `F`에는 범주형 상호작용에 강한 CatBoost를 75% 결합한다.
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
| `F` | 퓨처스리그·2군 경기 | 30,010 | exact LightGBM 쪽으로 75% 이동한다. |

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

퓨처스리그는 1군 정규시즌과 분포가 다르고 최근 regime의 영향이 컸다. 과거 전이 검증에서 exact LightGBM의 개선이 강하고 일관됐기 때문에 이 영역에만 높은 가중치를 사용한다.

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

10위까지는 `4.9158`, 1200까지는 `46.9564`점이 남아 있고 확인 당시 12위다. v22는 로컬 최소 `+4.6215` 중 Public `+1.5298`이 전이됐으므로 신호는 생존했지만, 사전 등록 규칙에 따라 같은 보정 계열의 사후 가중치 조정은 금지한다.

우선순위는 다음과 같다.

1. v22를 고정 incumbent로 두고 독립적인 신호만 nested rolling-origin에서 비교한다.
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

## Git·LFS 산출물 정책

다음 파일은 private 저장소라도 일반 Git이나 Git LFS에 포함하지 않는다.

```text
data/ 원본 파일
model/ 임시 모델 파일
artifacts/ allowlist 이외 OOF와 학습 산출물
output/ 예측 결과
submit*.zip
submissions/**/*.zip
.env 및 API token
개인 쿠키, 인증서와 key 파일
```

현재 예외는 `artifacts/standalone_champion_1161/standalone_champion_1161.zip`과
`artifacts/oof_champion_1161/*.npz`이며 `.gitattributes`의 Git LFS pointer로만
커밋한다. OOF에는 target과 선수 ID가 있으므로 저장소를 공개하거나 비팀원을 초대하기
전에 반드시 제거해야 한다. 과거 계보는 [팀 전용 Google Drive]([private reference removed])를
`제한됨`으로 유지해 사용한다. 원본 DACON 데이터와 인증정보는 Drive에도 올리지 않는다.

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

## 2026-08-22 TrackMan 전수조사

- TrackMan 1,793,078행을 공식 train과 target 없이 정렬한 결과 1,217,598행
  (82.54%)이 강한 일대일 근거로 연결됐다.
- test에는 현재 투구 물리량·plate location·intended target·TrackMan ID가 없으므로,
  TrackMan은 origin 이전 시즌의 투수별 동결 profile로만 사용한다.
- 고차원 물리 profile, repeatability, TrackMan 구종-label 학생 모델은 시간축에서 기각했다.
- 단일 IVB 4~9월 후보 v70은 2024 재사용 축에서 양수였지만, recipe를 고정한 v74의
  2022 과거축에서 eta `0`으로 반증되어 최종 기각했다.
- 상세 근거와 1170 후속 경로: [`reports/trackman_deep_dive_20260822.md`](reports/trackman_deep_dive_20260822.md)

## 2026-08-17 인계 요약

- Public champion: `submit_v22.zip`, `1153.0436023798`
- 확인 당시 순위: 12위, 제출 ID는 공식 리더보드 응답에서 미제공
- 10위까지 / 목표 1200까지: `4.9158471793 / 46.9563976202`
- 채택: v21 + 저자유도 도메인 보정 + row-local 투수·타자 ASOF prior
- 운영 기준: v22 고정 비교, 동일 보정 계열 미세 조정 금지
- Public 근거: [`reports/v22_public_result_20260817.md`](reports/v22_public_result_20260817.md)
