# 챔피언 및 기반 산출물 인계 안내

현재 Public champion은 standalone `1161.2020600422`이며 전달 기준은
[`STANDALONE_CHAMPION.md`](STANDALONE_CHAMPION.md)의 단일 ZIP이다. 이 문서의 Drive
구조는 v26 이하 역사 계보를 재현할 때만 사용한다. v26(`1157.9736407889`, 제출 ID
`51773`)의 직접 부모는 v25, v25의 직접 부모는 v22다. 최신 파일·해시는
[`PROJECT_STATUS.md`](PROJECT_STATUS.md)와 [`../submissions/README.md`](../submissions/README.md)를
우선한다. v21·v22·v25·v27 ZIP과 `artifacts/v25_postbreak_anchor_20260817_01/`은 아직
로컬 private artifact이며 Drive 업로드 완료로 표시하지 않는다.

2026-08-22 v84는 `submit_v84_probe.zip` 이름으로 제출해 Public `1161.2020600422`를
확인했다. 현재 인계 파일은
`artifacts/standalone_champion_1161/standalone_champion_1161.zip`, SHA-256은
`C033FC38A5F9681E45B0BD2494359B8BD1387EE44E5F318C5B7A5547CFE6C4F7`이다.

현재 1161 모델과 선별 OOF evidence는 private Git LFS에서 관리한다. 아래 팀 전용
[Google Drive]([private reference removed])는
v26 이하 과거 계보 전용이다.

```text
투구 제구 성공 확률 예측 팀 폴더/
├─ v17_현재최고점_1093/
└─ v13_과거기준선_1068/
```

v19 직접 부모가 필요하면 [v17 보관 폴더]([private reference removed])를 사용한다. [v13 과거 기준선 폴더]([private reference removed])는 v11→v13 계보를 분석하거나 과거 결과를 재현할 때만 사용한다.

## 딱 이렇게 선택한다

| 목적 | 받을 폴더·파일 | 다음 행동 |
|---|---|---|
| 최고점 실행·비교 | `v17.../01_최고점_실행/submit_v17.zip` | ZIP을 수정하지 않고 검증기의 incumbent로 사용 |
| v17 재패키징 | 위 파일 + `v17.../02_v17_다시만들기/` 전체 | `submit_v14.zip`은 루트에 두고 artifact ZIP은 루트에 압축 해제 |
| pressure EB 재학습 | 위 파일 전체 + `v17.../03_모델_재학습/` 전체 | training ZIP을 루트에 압축 해제한 뒤 워크벤치의 학습 단계 실행 |
| 과거 v13 연구 | `v13_과거기준선_1068/` | 폴더 안 `처음이라면_이것부터.txt`부터 확인 |

원본 `train.csv`, `test.csv`, `trackman_history.csv`, `sample_submission.csv`는 Drive에 올리지 않는다. 각 팀원이 DACON에서 직접 받아 프로젝트의 `data/`에 둔다.

## v17 현재 최고점 폴더

v17은 2026-08-16 Public `1093.3213473808`을 기록한 당시 챔피언이며 현재는 과거 계보 자료다.

```text
v17_현재최고점_1093/
├─ 00_처음이라면_여기부터/
│  ├─ 처음이라면_이것부터.txt
│  └─ manifest.json
├─ 01_최고점_실행/
│  └─ submit_v17.zip
├─ 02_v17_다시만들기/
│  ├─ submit_v14.zip
│  └─ v17_residual_artifacts.zip
└─ 03_모델_재학습/
   └─ v17_training_oof_minimal.zip
```

| 파일 | 크기(바이트) | SHA-256 | 역할 |
|---|---:|---|---|
| `submit_v17.zip` | 27,247,106 | `A327740E5AB995F48832B0D514EC7330F5B73EDDCD81407E8BE3326385CCD595` | Public 챔피언 실행·비교 |
| `submit_v14.zip` | 27,213,435 | `AC8E135FBA8DBF41E331A8F4E2AAA658F0AF6675DD5F3B6C89EF51E9BC02977E` | v17 재패키징 부모 |
| `v17_residual_artifacts.zip` | 34,668 | `BAC1C9858C491C06AD48C9A439E2C14A88627E65E9439EEAB93322DDDB0A7BF7` | 선택된 pressure EB spec과 manifest |
| `v17_training_oof_minimal.zip` | 49,715,429 | `AAAC5042E11C82240115FD4B3584A1A68E461FAD5970E3FA969EA9223A724A45` | v17 재학습에 필요한 최소 OOF cache |
| `처음이라면_이것부터.txt` | 2,467 | `0F0ED4763CB160FC8D3B8FC438FB3F9DEE3AA6391C34A084EDF2925CF2B2374F` | 비개발자용 시작 안내 |
| `manifest.json` | 2,770 | `68169BEC52748223DA261E4C54A43D299363C2605F51FE2A349A413A7689D111` | 전달 파일 경로·크기·해시 명세 |

artifact ZIP은 프로젝트 루트에서 압축을 풀어야 내부의 `artifacts/...` 경로가 올바르게 만들어진다.

```powershell
Expand-Archive -LiteralPath .\v17_residual_artifacts.zip -DestinationPath . -Force
Expand-Archive -LiteralPath .\v17_training_oof_minimal.zip -DestinationPath . -Force
```

`submit_v17.zip`과 `submit_v14.zip`은 압축을 풀거나 다시 압축하지 않는다.

## v13 과거 기준선 폴더

v13은 Public `1068.4365711741`을 기록한 역사적 기준선이다. 새 실험의 기본 출발점은 v17이며, 아래 파일은 v13 overlay 분해·재현이 필요할 때만 받는다.

```text
v13_과거기준선_1068/
├─ 00_처음이라면_여기부터/
│  ├─ 처음이라면_이것부터.txt
│  └─ manifest.json
├─ 01_기준점_실행/
│  └─ submit_v13_fixed.zip
├─ 02_v13_다시만들기/
│  ├─ submit_v11.zip
│  └─ v13_overlay_artifacts.zip
└─ 03_모델_재학습/
   ├─ v13_frozen_oof.zip
   └─ 참고용_압축해제본/
```

| 파일 | 크기(바이트) | SHA-256 | 역할 |
|---|---:|---|---|
| `submit_v13_fixed.zip` | 27,164,737 | `D8CB028806223610A8F61AF961CDFBA11B50282D08A029F4C285D732E10E1283` | v13 기준 제출 패키지 |
| `submit_v11.zip` | 27,218,745 | `F5D0C9DD677F27EC586735E5645CFF9E84337047113AB5D6695CD59EB6B27F0D` | v13 재패키징 부모 |
| `v13_overlay_artifacts.zip` | 220,954 | `1678A3DFC38E74ACCDF77D3409C5D52F94DEE996CB12EB20C66125E9C69964EB` | exact-ASOF overlay 모델·전처리·설정 |
| `v13_frozen_oof.zip` | 7,344,908 | `0834B3C04FAE6C7AF413A3DA2FEAC37276A4AE27AAFCCA5CB9635928A32828D6` | v13 overlay 재학습용 최소 OOF |
| `처음이라면_이것부터.txt` | 1,796 | `4FE2351B45FE1E5F1A6A7EA5A3D843C7F4B37D878F2AE67BF151DF2907219397` | 과거 기준선 안내 |
| `manifest.json` | 2,336 | `A4ED05430601FA0C247609A4C7CA370E0E032622B3D23DBADE9EFF9580C7463D` | 전달 파일 경로·크기·해시 명세 |

`참고용_압축해제본/`은 v13 OOF 세 파일을 눈으로 확인하기 위한 사본이다. 재학습 코드는 경로 구조가 보존된 `v13_frozen_oof.zip`을 프로젝트 루트에 푸는 방식을 기준으로 한다.

## 해시 확인

다운로드 직후 프로젝트 루트에서 필요한 파일만 확인한다.

```powershell
Get-FileHash -Algorithm SHA256 .\submit_v17.zip
Get-FileHash -Algorithm SHA256 .\submit_v14.zip
Get-FileHash -Algorithm SHA256 .\v17_residual_artifacts.zip
Get-FileHash -Algorithm SHA256 .\v17_training_oof_minimal.zip
```

표의 값이나 각 폴더의 `manifest.json`과 다르면 실행하지 말고 다시 내려받는다. 현재 파일 이동·크기·Drive readback 검증은 완료됐으며, 별도 팀원 계정으로 내려받은 뒤의 독립 SHA-256 재확인만 남아 있다.

## 실행 진입점

팀원이 Python 스크립트를 하나씩 찾아 실행할 필요는 없다. 저장소 루트에서 아래 노트북을 연다.

```powershell
jupyter lab notebooks/experiment_workbench.ipynb
```

[`../notebooks/experiment_workbench.ipynb`](../notebooks/experiment_workbench.ipynb)가 현재 공식 단일 실행 노트북이다. 경로 확인, 테스트, screen, 강건 평가, 학습, 패키징과 검증 모듈을 순서대로 호출한다. 처음에는 `DRY_RUN=True`, `RUN_HEAVY=False`, `RUN_PACKAGING=False`를 유지한다. 상세 절차는 [`EXPERIMENT_WORKFLOW.md`](EXPERIMENT_WORKFLOW.md)에 있다.

## 공유·보안 원칙

- Drive의 `일반 액세스`는 `제한됨`으로 유지하고 공식 DACON 팀원 계정만 개별 초대한다.
- PRIVATE 팀 전달용 `submissions/releases/v167/submit_v167.zip`, 기존 1161 allowlist와
  해시가 고정된 `artifacts/oof_champion_1170/` 번들 이외의 모델·OOF·제출 ZIP은 Git
  commit이나 Git LFS에 넣지 않는다.
- LFS OOF에는 target과 선수 ID가 있으므로 저장소는 private·공식 팀원 3인으로 유지한다.
- DACON 원본 데이터와 토큰·쿠키·키 같은 인증정보는 Drive에도 올리지 않는다.
- 파일을 교체할 때 기존 버전을 덮어쓰지 않고 새 버전 폴더와 manifest를 만든다.
- GitHub Issue와 Pull Request에는 긴 파일 목록 대신 이 문서, 폴더 버전과 SHA-256을 적는다.
