# v335 Public 1181.7100 standalone release

## 확정 파일

| 항목 | 값 |
|---|---:|
| 파일 | `submit_v335_anchor_lowrank_complement.zip` |
| DACON 업로드명 | `submit_v335.zip` |
| 제출 ID | `76835` |
| 제출 시각 | `2026-08-31 00:00:45 KST` |
| Public | **1181.7100031613** |
| 공식 runtime | `111초` |
| SHA-256 | `A3BCD933DE3F78DEB9565F537DC3199922BDFD219B2840E4150F939CFF144C7A` |
| 크기 / 멤버 | `91,469,892 bytes` / `188` |

이 ZIP은 루트 `script.py`, `requirements.txt`, `model/`만으로 실행되는 독립 패키지다.
공식 `data/test.csv`와 `data/sample_submission.csv`만 실행 폴더에 제공하면 과거 제출 ZIP이나
저장소 소스를 참조하지 않는다.

## v290과의 차이

v335는 v290의 R_CORE 예측을 그대로 보존하면서 두 개의 고정된 저용량 보완을 추가한다.

- `F`: 최근 퓨처스 direct expert를 10%에서 20%로 확대하고, 과거 forward OOF에서 고정한
  `lowrank_s300_r2` 투수×카운트×타자손 상호작용을 0.50 강도로 추가한다.
- `R_ANCHOR`: 정규시즌에서 `pitcher_team_id == 13` 또는 `batter_team_id == 13`인 행에
  동일한 finalized low-rank 상호작용을 0.50 강도로 추가한다.
- `R_CORE`: v290과 수치적으로 동일하다.

중간 후보 v320은 제출하지 않았으므로 Public `+4.9529314933`을 F와 R_ANCHOR에 각각
분해해 귀속하지 않는다. 상세 검증과 제한은
[`../../../reports/v335_public_result_20260831.md`](../../../reports/v335_public_result_20260831.md)에 있다.

## 실행

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

생성되는 `output/submission.csv`의 행 ID는 sample submission 순서를 따른다.
