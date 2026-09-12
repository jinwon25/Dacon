# 제출 ZIP 파일명 및 제출 이력 규칙

## 고정 규칙

- 최종 제출 ZIP 파일명은 확장자를 포함해 40자 이내로 한다.
- 파일명은 제출 버전을 즉시 확인할 수 있도록 `submit_vN.zip` 형식을 기본으로 한다.
- 모델 구성·실험명은 `artifacts/candidates/*/manifest.json`과 보고서에 기록하고, 최종 파일명에는 넣지 않는다.
- 다음 사용 가능한 제출은 `submit_v19.zip`이다. 기존 제출 파일은 덮어쓰지 않는다. 새 ZIP은 고정 promotion gate와 팀 리뷰를 통과할 때만 만든다.
- ZIP, 모델과 OOF는 원칙적으로 Git에 커밋하지 않는다. PRIVATE 팀 전달용으로 승인된
  `submissions/releases/v167/submit_v167.zip`과 문서화된 기존 1161 LFS·1170 OOF
  allowlist만 해시가 고정된 예외다.
- 같은 버전의 재현 검사용 로컬 출력은 `submit_v17_rebuild.zip`처럼 목적을 붙이고 DACON 제출 파일과 구분한다.

## 최근 의사결정 기록

| 버전/제목 | 제출 ID | 제출 일시 | 제출 선택 | 점수 | 소요 시간 | 파일명 |
|---|---:|---|---|---:|---:|---|
| v13 exact-ASOF | 1532352 | 2026-08-15 11:11:53 | schedule | 1068.4365711741 | - | `submit_v13_fixed.zip` |
| submission17 edit | 50033 | 2026-08-16 00:01:32 | schedule | **1093.3213473808** | 16초 | `submit_v17.zip` |
| submission18 edit | 50048 | 2026-08-16 00:03:21 | - | 1090.4672420401 | - | `submit_v18.zip` |

v17은 v13보다 `+24.8847762067` 올라 새 챔피언이 됐다. 같은 pressure EB를 쓰면서 F trend를 더 공격적으로 확대한 v18은 v17보다 `-2.8541053407` 낮았으므로 해당 확장은 기각한다. 전체 제출 이력과 가설·결론은 [`submissions.csv`](submissions.csv)를 단일 원본으로 사용한다.
