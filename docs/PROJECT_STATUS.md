# 프로젝트 현황

이 문서는 처음 참여한 팀원이 현재 champion, 재현 경로와 다음 작업 규칙을 빠르게 파악하기 위한 기준 문서다. 마지막 갱신은 `2026-08-16 KST`다.

## 현재 champion

| 항목 | 값 |
|---|---:|
| 목표 Public | **1160** |
| 현재 최고 Public | **1151.472428719** |
| 목표까지 차이 | **8.5275712810** |
| champion | `submit_v20.zip` |
| 제출 ID | `1534122` |
| 제출 시각 | `2026-08-16 15:47:28 KST` |
| 확인 당시 순위 | **12위** |
| SHA-256 | `4E50A6B7C7D970AF9B8A3656C4D87F2A7D856321C53416E831C4B8E5B01BAE29` |
| API 응답 | `isSubmitted=true`, `detail=Success` |

공식 결과의 단일 근거는 [`../reports/v20_public_result_20260816.md`](../reports/v20_public_result_20260816.md), 전체 제출 이력의 단일 기준표는 [`../reports/submissions.csv`](../reports/submissions.csv)다.

## 최근 Public 의사결정

| 후보 | Public | v17 대비 | 결정 |
|---|---:|---:|---|
| v13 | 1068.4365711741 | -24.8847762067 | exact-ASOF 역사 기준선 |
| v17 | 1093.3213473808 | 기준 | v19 직접 부모 |
| v18 | 1090.4672420401 | -2.8541053407 | 공격적 F 강화 기각 |
| v19 | 1144.1518063753 | +50.8304589945 | v20 직접 부모 |
| **v20** | **1151.472428719** | **+58.1510813382** | 현재 champion |

## champion 계보

```text
v11 공개 검증 앙상블
  └─ v13 exact-ASOF overlay
       └─ v14 보수적 domain refinement
            └─ v17 R_CORE pressure empirical-Bayes residual
                 └─ v19 multi-year row-local state ensemble
                      + latent failure-mode domain routing
                      └─ v20 pooled residual lookup + PFD student delta
```

v20은 v19의 예측을 부모로 두고 강하게 수축한 선수·팀·카운트 잔차 lookup, latent failure-mode와 training-only TrackMan PFD student delta를 결합한다. `R_CORE`, `R_ANCHOR`, `F`마다 보호 규칙을 사용하며 다른 test 행의 분포·빈도·순서에는 의존하지 않는다.

## 활성 파일

### 바로 읽을 문서

```text
README.md
submissions/README.md
reports/README.md
reports/v20_public_result_20260816.md
reports/target1160_v20_candidate_20260816.md
docs/REFERENCE_MAP.md
```

### v20 구현·검증

```text
src/train_v20_target1160.py
src/package_v20_target1160.py
src/validate_v20_target1160.py
src/v10_overlay_script.py
artifacts/v20_target1160_final_20260816/manifest.json
reports/v20_validation.json
```

### 기본 입력 위치

```text
submit_v20.zip
submissions/history/submit_v19.zip
data/train.csv
data/test.csv
data/trackman_history.csv
data/sample_submission.csv
```

원본 데이터, 모델, OOF와 제출 ZIP은 Git에서 제외한다. 과거 ZIP의 위치와 보존 정책은 [`../submissions/README.md`](../submissions/README.md)를 따른다.

## 검증 결과

- 세 순방향 검증축 모두 v19 대비 양수
- 패키지 게이트: 12개 모두 통과
- 2023→2024 BSS-equivalent gain vs v19: `+14.251663`
- 245,789행 추론: `93.274초`
- peak RSS: `1,434.9MiB`
- 분할 배치 최대 절대 차이: `1.110e-16`
- Public gain vs v19: `+7.3206223437`

## 작업 원칙

1. 팀 공통 저장소는 `https://github.com/Lg-Aimers-chungang/hackathon`이다.
2. `main`에 직접 push하지 않고 개인 feature branch에서 작업한다.
3. v20을 덮어쓰지 않고 새 후보 파일명을 사용한다.
4. 평가 시즌보다 과거 데이터만 학습·보정에 사용한다.
5. 테스트 배치의 평균·빈도·순서 등 다른 test 행 정보는 사용하지 않는다.
6. Public 결과에 맞춘 사후 weight 미세 조정은 하지 않는다.
7. ZIP 계보, offline 추론, 120초 제한, 메모리와 배치 불변성을 확인한다.
8. 제출 후 `reports/submissions.csv`와 별도 Public 결과 문서를 함께 갱신한다.

## 다음 연구 우선순위

1160까지 `8.5275712810`점이 남아 있고 확인 당시 12위다. 다음 후보는 v20 고정 비교를 전제로 한다.

1. v20과 상관이 낮고 시간 전이에서 독립적으로 양수인 신호를 찾는다.
2. `R_CORE`, `R_ANCHOR`, `F` 변경을 한 후보 안에서 섞지 않고 영역별로 분리한다.
3. 2022→2023, 2023→2024 forward fold와 월·투수·타자·pitch-block 의존성 검증을 유지한다.
4. v18에서 기각된 공격적 F 확장은 재사용하지 않는다.
5. 남은 1회 제출은 로컬 게이트와 팀 리뷰를 통과한 후보에만 사용한다.

현재 운영 결정은 v20 champion 고정이다.
