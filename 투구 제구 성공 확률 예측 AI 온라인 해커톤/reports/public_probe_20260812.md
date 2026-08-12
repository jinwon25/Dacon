# Public probe audit — submission 45330

## Result

| item | value |
|---|---:|
| submission | `45330` (`submission7 edit`) |
| submitted at | `2026-08-12 22:00:33` |
| package | `submit_game_type.zip` |
| package SHA-256 | `99C8A68AAE4095162A018FD3DDB0DA434BE27DE12CEAB7574C7C49F153078A58` |
| Public score | **754.3580670546** |
| parent `submit_v2.zip` | **763.2665303697** |
| delta vs parent | **-8.9084633151** |
| runtime | 9 seconds |

## Decision

The candidate improved every cached 2021–2024 walk-forward fold, but the untouched Public evaluation rejected the aggressive game-type Trackman routing. It is therefore **not promoted**. `submit_v2.zip` remains the verified public champion and should be used for the next submission unless a new candidate passes both the temporal OOF gate and a future public probe.

The result is recorded in `reports/submissions.csv` and the candidate manifest. The candidate ZIP is retained for reproducibility; no model files were overwritten.
