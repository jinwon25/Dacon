# 최신 단일 실행 챔피언

마지막 갱신: `2026-08-31 KST`

## 확정 릴리스

| 항목 | 값 |
|---|---:|
| Public | **1181.7100031613** |
| 제출 ID | `76835` |
| DACON 업로드명 | `submit_v335.zip` |
| 파일 | `submissions/releases/v335/submit_v335_anchor_lowrank_complement.zip` |
| SHA-256 | `A3BCD933DE3F78DEB9565F537DC3199922BDFD219B2840E4150F939CFF144C7A` |
| 크기 / 파일 수 | `91,469,892 bytes` / `188` |
| 공식 runtime | `111초` |
| 루트 | `script.py`, `requirements.txt`, `model/` |

v335는 이전 챔피언 v290 `1176.757071668`보다 `+4.9529314933` 개선됐고 목표 1180을
`+1.7100031613` 상회했다. 현재 실행·전달·검증 기준은 이 ZIP이다.

## 독립 실행 계약

최신 ZIP은 과거 제출 ZIP, 저장소 `src/`, 학습 코드나 외부 모델 경로를 요구하지 않는다.
모든 추론 코드는 `script.py`, 모든 모델과 고정 통계는 `model/`에 포함돼 있다. 필요한 외부
입력은 공식 `test.csv`와 `sample_submission.csv`뿐이다.

```text
run_v335/
├─ script.py
├─ requirements.txt
├─ model/
├─ data/
│  ├─ test.csv
│  └─ sample_submission.csv
└─ output/
   └─ submission.csv
```

### 1. 파일 무결성 확인

```powershell
Get-FileHash -Algorithm SHA256 `
  submissions/releases/v335/submit_v335_anchor_lowrank_complement.zip
```

결과가
`A3BCD933DE3F78DEB9565F537DC3199922BDFD219B2840E4150F939CFF144C7A`와 정확히 같아야 한다.

### 2. 깨끗한 폴더에서 실행

```powershell
$release = Join-Path $PWD "run_v335"
New-Item -ItemType Directory -Force $release | Out-Null
Expand-Archive `
  submissions/releases/v335/submit_v335_anchor_lowrank_complement.zip `
  $release -Force
New-Item -ItemType Directory -Force (Join-Path $release "data") | Out-Null
Copy-Item data/test.csv (Join-Path $release "data/test.csv")
Copy-Item data/sample_submission.csv (Join-Path $release "data/sample_submission.csv")
python -m pip install -r (Join-Path $release "requirements.txt")
python (Join-Path $release "script.py")
```

결과는 `run_v335/output/submission.csv`에 생성된다. 원본 데이터와 출력은 Git에 추가하지 않는다.

### 3. 저장소 감사기로 재검증

```powershell
python -m src.archive.audit_standalone_release `
  --package submissions/releases/v335/submit_v335_anchor_lowrank_complement.zip `
  --test-csv data/test.csv `
  --scale-rows 245789 `
  --timeout-seconds 300
```

감사기는 ZIP CRC·루트 구조·금지된 배치 집계·출력 스키마·확률 범위·셔플/분할 행 독립성·
전체 규모 runtime을 검사한다.

## 모델 변경과 재감사 결과

v335는 v290 계보의 R_CORE를 보존하고 다음 두 disjoint 보완을 추가한다.

- `F`: 최근 direct expert 총 20% + finalized `lowrank_s300_r2` 0.50
- `R_ANCHOR`: team 13 연관 정규시즌 행에 같은 low-rank 0.50
- `R_CORE`: v290과 동일

| 검사 | 결과 |
|---|---:|
| ZIP SHA / bytes / 파일 수 | 일치 |
| ZIP CRC | 통과 |
| v320 대비 F parity | `0.0` |
| v320 대비 R_CORE parity | `0.0` |
| R_ANCHOR 공식 최대 오차 | `0.0` |
| shuffle / partition | `0.0` / `5.55e-17` |
| 확률 유한성·범위 | 통과 |
| DACON 공식 runtime | `111초` |
| DACON 공식 제한 | `600초` |

중간 F-only v320은 제출하지 않았으므로 Public `+4.9529314933`을 F와 R_ANCHOR에 분해해
귀속하지 않는다. 자세한 판단은
[`../research/reports/v335_public_result_20260831.md`](../research/reports/v335_public_result_20260831.md)에 있다.

## 실행과 재구축의 차이

- **실행**: v335 ZIP과 공식 test/sample만 필요하다.
- **재검증**: 위 항목에 저장소 감사 코드가 추가로 필요하다.
- **재구축**: 공식 train/forward OOF, v290 부모 ZIP과 v319 low-rank lookup이 필요하다.

핵심 진입점:

```powershell
python -m src.champion.v319_finalize_futures_lowrank --help
python -m src.champion.v320_build_futures_portfolio_package --help
python -m src.archive.v335_anchor_lowrank_complement_audit --help
python -m src.champion.v335_build_anchor_lowrank_package --help
python -m src.archive.audit_v335_anchor_lowrank --help
```

재구축 결과는 기존 챔피언을 덮어쓰지 않고 새 디렉터리에 만든 뒤 SHA·수식 parity·행 독립성·
runtime을 모두 확인한다.

## 전달 정책

이 저장소가 PRIVATE임을 확인한 뒤, 최신 v335 ZIP을 해시 고정 예외로 추적한다. 공식 DACON
데이터, 인증정보, 부모 ZIP, 일반 모델·OOF는 계속 제외한다. 기존 v167·JY 릴리스와 1161
LFS·1170 OOF allowlist는 역사적 재현 자산으로 보존한다.

팀 전달 시 함께 확인할 항목:

1. `submissions/releases/v335/submit_v335_anchor_lowrank_complement.zip`
2. 위 SHA-256
3. 이 문서와 [`../research/reports/v335_public_result_20260831.md`](../research/reports/v335_public_result_20260831.md)
