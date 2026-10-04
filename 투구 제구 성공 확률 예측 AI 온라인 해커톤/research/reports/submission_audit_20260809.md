# Submission audit 2026-08-09

`reports/submissions.csv`의 v3–v5 행에는 v2의 전체 walk-forward Brier `0.2476249274184975`가 반복 기록되어 있었다. 이는 각 후보를 동일한 기준으로 재평가한 candidate-specific local Brier가 아니므로 이번 사이클에서 빈 값으로 교정한다. Public score와 ZIP SHA는 원래 기록을 보존한다.

| version | prior local_brier | corrected local_brier | reason |
|---|---:|---:|---|
| v3 | 0.2476249274184975 | NA | Trackman 10% 후보의 독립 local fold Brier가 원장에 없음 |
| v4 | 0.2476249274184975 | NA | R-only 후보의 독립 local fold Brier가 원장에 없음 |
| v5 | 0.2476249274184975 | NA | R-only+Trackman 후보의 독립 local fold Brier가 원장에 없음 |

이전 값은 추정·복사해 대체하지 않았으며, 제출 Public 점수는 `submission_audit`와 `submission_naming`에 그대로 유지했다. 다음 유효 파일명은 사용자 기준에 따라 `submit_v6.zip`으로 고정한다.
