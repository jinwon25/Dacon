# 1161.2020600422 단일 실행 챔피언

> **최신 운영 기준 변경(2026-08-22 23:06 KST)**: v104가 Public
> **1162.6302840289**(제출 ID `60626`)를 기록해 현재 전달 파일은
> `artifacts/standalone_champion_1162/standalone_champion_1162.zip`이다. 아래 본문은
> 직전 v84/1161 패키지의 상세 감사 기록으로 보존한다. 최신 근거와 실행 방법은
> [`../reports/target1170_v104_public_result_20260822.md`](../reports/target1170_v104_public_result_20260822.md)를 따른다.

## 목적

현재 단일 릴리스는 직전 1159 챔피언과 v56 shared FM을 한 ZIP 안에 완전히
포함한다. 실행 시 과거 제출 ZIP, 저장소 `src/`, 외부 모델 경로를 요구하지 않는다.

공식 Public은 `1161.2020600422`다. v82의 R_CORE strict·R_ANCHOR TrackMan 계보를
그대로 보존하고, source-period-balanced rank-16 pairwise FM correction을 F에서 logit
10% 적용한 v84다.

현재 확정 ZIP은 `artifacts/standalone_champion_1161/standalone_champion_1161.zip`,
SHA-256은 `C033FC38A5F9681E45B0BD2494359B8BD1387EE44E5F318C5B7A5547CFE6C4F7`,
크기는 `34,629,670` bytes다. `standalone_manifest.json`에 Public 결과와 71개 파일,
smoke/compliance 결과가 기록되어 있다.

private 저장소를 clone한 공식 팀원은 먼저 LFS payload를 받는다. OOF evidence는 실행에
필요하지 않으며 후속 비교·감사용이다.

```powershell
git lfs install
git lfs pull --include="artifacts/standalone_champion_1161/*,artifacts/oof_champion_1161/*"
```

## 현재 1161 ZIP 재검증

아래 명령은 ZIP을 임시 디렉터리에 풀고 저장소 모델이나 과거 제출물을 참조하지 않은
상태로 구조·CRC·출력 스키마·확률 범위·행 독립성을 다시 검사한다.

```powershell
python -m src.audit_standalone_release `
  --package artifacts/standalone_champion_1161/standalone_champion_1161.zip `
  --test-csv data/test.csv `
  --scale-rows 245789 `
  --timeout-seconds 180
```

모델 계보와 OOF evidence까지 함께 검사하려면 공식 `train.csv`를 준비하고 실행한다.

```powershell
python -m src.audit_champion_private_artifacts `
  --model-zip artifacts/standalone_champion_1161/standalone_champion_1161.zip `
  --model-manifest artifacts/standalone_champion_1161/standalone_manifest.json `
  --oof-dir artifacts/oof_champion_1161 `
  --train-csv data/train.csv
```

2026-08-22 실물 ZIP 재검증에서는 정적 금지 연산 0건, 5행의 단일행·순서변경·분할
예측 최대 차이 각각 `1.11e-16`, `0`, `0`을 확인했다. 245,789행 격리 재실행은
현재 PC 부하에서 `114.487초`였고 모든 확률이 유한하며 `[0, 1]` 범위였다. 실제 DACON
제출 실행시간은 `41초`로 공식 제한 `600초` 이내다.

`src.package_standalone_champion`의 기본 프로토콜과 메타데이터는 1158 역사 패키지
복원용이다. 현재 1161 릴리스의 일반 재검증에는 위 `src.audit_standalone_release`를
사용한다.

## 1158 역사 패키지 materialize

팀원의 최종 제출 ZIP이 보존되지 않아, 병합된 고정 레시피로 복원한 최종 패키지를
한 번만 입력한다.

```powershell
python -m src.package_standalone_champion `
  --source-package champion_1158_source.zip `
  --payload-dir artifacts/standalone_champion_1158/payload `
  --output artifacts/standalone_champion_1158/standalone_champion_1158.zip `
  --smoke-test-csv data/test.csv
```

생성물은 다음과 같다.

```text
artifacts/standalone_champion_1158/
├─ payload/
│  ├─ model/
│  ├─ requirements.txt
│  └─ script.py
├─ standalone_manifest.json
└─ standalone_champion_1158.zip
```

## 1158 역사 패키지 재패키징

payload가 만들어진 뒤에는 `--source-package`가 필요 없다.

```powershell
python -m src.package_standalone_champion `
  --payload-dir artifacts/standalone_champion_1158/payload `
  --output artifacts/standalone_champion_1158/standalone_champion_1158.zip `
  --smoke-test-csv data/test.csv
```

출력 ZIP이 이미 있으면 덮어쓰지 않는다. 기존 파일을 보존하거나 명시적으로 다른
출력명을 사용한 뒤, 검증을 통과한 한 파일만 `standalone_champion_1158.zip`으로
확정한다.

## DACON 평가 행 독립성 검증

[공식 재안내](https://dacon.io/competitions/official/236743/talkboard/417123)의
핵심 원칙은 평가 데이터 각 행을 독립적으로 예측하는 것이다. 특정 행은 그 행의
입력과 공식 학습 데이터로 미리 고정한 모델·통계만 사용할 수 있다. 같은
`test.csv`의 다른 행으로 만든 누적·rolling/lag·평균·분포·빈도·순위·선수/팀/월/
경기 집계는 금지된다.

이 빌더는 두 단계로 이를 검사한다.

1. 빌드 전 `script.py`를 정적으로 검사해 `groupby`, `rolling`, `shift`, `rank`,
   `quantile`, `value_counts`, 누적 연산 등 고위험 평가 배치 연산을 거부한다.
2. `--smoke-test-csv` 또는 `--compliance-test-csv`를 주면 같은 행의 예측을
   전체 배치, 한 행씩, 셔플 배치, 분할 배치에서 비교한다. 최대 절대 차이가
   `1e-12`를 넘으면 패키지를 통과시키지 않는다.

현재 챔피언의 프로필 조인은 `model/`에 저장된 공식 학습 데이터 기반 고정
프로필만 사용한다. 조인 뒤의 정렬은 입력 순서 복원용이며, 평가 배치의 통계나
순위를 예측값에 사용하지 않는다.

## 전달받은 ZIP 실행

ZIP을 푼 디렉터리에 공식 `data/test.csv`를 놓고 아래처럼 실행한다.

```powershell
python -m pip install -r requirements.txt
python script.py
```

결과는 `output/submission.csv`에 생성된다. `script.py`는 `src` 또는 과거 버전
코드를 import하지 않으며, 모든 추론 모델은 같은 ZIP의 `model/`에서만 읽는다.

## 전달 묶음

팀 인계 시에는 아래 세 항목만 전달한다.

1. `standalone_champion_1161.zip`
2. `standalone_manifest.json`
3. 이 문서

원본 DACON 데이터와 개인 인증정보는 전달 묶음에 포함하지 않는다.

## v82 제출과 승격

동일 해시 파일을 `submit_v82_probe.zip` 이름으로 제출해 API 성공을 확인했고, Public
`1159.3352239501`로 직전 챔피언보다 `+1.2606682750` 상승했다. 현재 로컬 전달명만
`standalone_champion_1159.zip`으로 정리했으며 ZIP 내용과 해시는 제출본과 동일하다.
상세 감사는
[`../reports/target1170_v82_public_probe_20260822.md`](../reports/target1170_v82_public_probe_20260822.md)에 있다.

## v84 제출과 승격

v83에서 사전 고정한 shared FM 레시피를 full-2023+late-2024 source로 재학습했다.
제출 파일 `submit_v84_probe.zip`은 API 성공 후 Public `1161.2020600422`를 기록해
정확한 v82 점수 `1159.3352239501`보다 `+1.8668360921` 상승했다. 제출 이력 ID는
`59988`이다.

제출본은 과거 ZIP을 실행 시 참조하지 않는다. v82 전체 payload를 `model/parent/`에,
NumPy용 FM vocabulary·embedding·center를 `model/v56_fm/`에 포함한다. canonical 이름만
`standalone_champion_1161.zip`으로 바꿨으며 바이트와 SHA-256은 제출본과 동일하다.
상세 감사는
[`../reports/target1170_v84_public_result_20260822.md`](../reports/target1170_v84_public_result_20260822.md)에 있다.
