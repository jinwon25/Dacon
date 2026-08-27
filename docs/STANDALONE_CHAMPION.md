# v167 단일 실행 챔피언

마지막 갱신: `2026-08-27 KST`

## 확정 릴리스

| 항목 | 값 |
|---|---:|
| Public | **1172.0772380321** |
| 순위 / 제출 ID | **11위** (확인 시점) / `1547707` |
| 파일 | `submissions/releases/v167/submit_v167.zip` |
| SHA-256 | `30DD28F56723EC0F560C9101FC5A94EF78568F874DCA88BF808879831E61C8C1` |
| 크기 / 파일 수 | `46,354,018 bytes` / `89` |
| 루트 | `script.py`, `requirements.txt`, `model/` |

v167은 v148 대비 Public `+1.7757683144`를 기록했다. 현재 실행·전달·검증 기준은
v167 하나다.

## 독립 실행 계약

v167 ZIP은 과거 제출 ZIP, 저장소 `src/`, 학습 코드나 외부 모델 경로를 요구하지 않는다.
모든 추론 코드는 `script.py`, 모든 모델과 고정 통계는 `model/`에 포함돼 있다. 실행 시
필요한 외부 입력은 공식 `test.csv`와 `sample_submission.csv`뿐이다.

```text
run_v167/
├─ script.py
├─ requirements.txt
├─ model/
├─ data/
│  ├─ test.csv
│  └─ sample_submission.csv
└─ output/               # 실행 시 생성
   └─ submission.csv
```

### 1. 파일 무결성 확인

```powershell
Get-FileHash -Algorithm SHA256 `
  submissions/releases/v167/submit_v167.zip
```

결과가
`30DD28F56723EC0F560C9101FC5A94EF78568F874DCA88BF808879831E61C8C1`와 정확히 같아야 한다.

### 2. 깨끗한 폴더에서 실행

```powershell
$release = Join-Path $PWD "run_v167"
New-Item -ItemType Directory -Force $release | Out-Null
Expand-Archive `
  submissions/releases/v167/submit_v167.zip `
  $release -Force
New-Item -ItemType Directory -Force (Join-Path $release "data") | Out-Null
Copy-Item data/test.csv (Join-Path $release "data/test.csv")
Copy-Item data/sample_submission.csv (Join-Path $release "data/sample_submission.csv")
python -m pip install -r (Join-Path $release "requirements.txt")
python (Join-Path $release "script.py")
```

결과는 `run_v167/output/submission.csv`에 생성된다. 원본 데이터와 결과 파일은 Git에
추가하지 않는다.

### 3. 저장소 감사기로 재검증

```powershell
python -m src.audit_standalone_release `
  --package submissions/releases/v167/submit_v167.zip `
  --test-csv data/test.csv `
  --scale-rows 245789 `
  --timeout-seconds 300
```

감사기는 임시 디렉터리에 ZIP을 풀고 ZIP 내부의 코드·모델만 실행한다. 다음 조건 중 하나라도
어기면 실패한다.

1. ZIP CRC 또는 허용 루트 구조가 잘못됐다.
2. 실행 스크립트에서 평가 배치 집계를 유발할 수 있는 금지 연산을 발견했다.
3. 같은 행의 단일행·셔플·분할 예측 차이가 `1e-12`를 넘는다.
4. 출력 ID, 길이, 유한성 또는 `[0, 1]` 확률 범위가 잘못됐다.
5. 지정한 timeout을 넘는다.

## 재감사 결과

2026-08-25 공식 test 사본으로 다시 검사한 결과다.

| 검사 | 결과 |
|---|---:|
| ZIP SHA / bytes / 파일 수 | 일치 |
| CRC | 통과 |
| 정적 금지 연산 | `0건` |
| 수식 parity 최대 오차 | `1.11e-16` |
| 비활성 경로 보존 최대 오차 | `1.11e-16` |
| shuffle vs 전체 최대 오차 | `1.11e-16` |
| partition vs 전체 최대 오차 | `1.11e-16` |
| 30행 mixed-route smoke | `5.544초`, 유한·범위 통과 |
| 245,789행 현재 부하 실행 | `88.498초`, 유한·범위 통과 |
| DACON 공식 제한 | `600초` |

저장소의 120초는 후보 비교용 내부 soft guard이며 공식 제한을 대체하지 않는다.

## 실행과 재구축의 차이

- **실행**: 확정 v167 ZIP과 공식 test/sample 두 파일만 있으면 된다.
- **재검증**: 위 항목에 저장소의 `src.audit_standalone_release`만 추가로 필요하다.
- **재구축**: 공식 train/OOF와 해시가 고정된 v104·H1 부모 artifact가 추가로 필요하다.

재구축 인수 계약은 다음 명령으로 확인한다.

```powershell
python -m src.archive.v167_build_h1_affine_submission_package --help
```

재구축 결과는 기존 챔피언을 덮어쓰지 않고 새 출력 디렉터리에 만든 뒤, SHA·수식 parity·
행 독립성·전체 런타임이 모두 통과할 때만 비교한다.

## 전달 정책

이 저장소가 PRIVATE임을 확인한 뒤, 팀 전달을 위해 검증된 v167 ZIP 한 개만
`submissions/releases/v167/submit_v167.zip`으로 추적한다. GitHub clone에는 이 확정 ZIP이
포함된다. 공식 DACON 데이터, 인증정보, 부모 ZIP과 승인되지 않은 다른 모델·OOF는 계속
Git에서 제외한다. 기존 1161 LFS·1170 OOF 번들은 문서화된 allowlist로 유지하고, 새 ZIP을
추가하려면 별도 승인과 SHA-256 기록이 필요하다. 팀 전달 시에는 다음 항목을 함께 확인한다.

1. `submissions/releases/v167/submit_v167.zip`
2. 위 SHA-256
3. 이 문서와 [`../reports/target1180_v167_public_result_20260825.md`](../reports/target1180_v167_public_result_20260825.md)

과거 `submit_v148.zip`, `standalone_champion_1161.zip`, `standalone_champion_1162.zip`과 v142 이하는 계보
감사용일 뿐 현재 실행 대상이 아니다.
