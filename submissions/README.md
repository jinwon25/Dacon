# 제출 파일 보관 규칙

현재 DACON Public 챔피언은 프로젝트 루트의 [`../submit_v26.zip`](../submit_v26.zip)이다. (2026-08-18 정정: 이전에는 `submit_v27.zip`을 champion으로 기록했으나 공식 제출 이력 재대조 결과 오류였다. 근거는 [`../reports/target1170_followup_20260817.md`](../reports/target1170_followup_20260817.md) 상단, [`../notebooks/v26_champion_reproduction.ipynb`](../notebooks/v26_champion_reproduction.ipynb) 참고.)

| 항목 | 값 |
|---|---:|
| Public 점수 | **1157.9736407889** |
| 확인 당시 순위 | 미확인 (정정 이전 "10위"는 v27 기준 기록) |
| 제출 ID | `51773` |
| SHA-256 | `8BE26E156A91C1FA5E9989EEE5043DB5992A960DF35A53D8A9BFEF7E6825D096` |
| v25 대비 | **+2.1443002480** |
| 목표 1170까지 | **12.0263592111** |
| 목표 1200까지 | **42.0263592111** |

## 제출하지 않을 동일 계열 후보

| 파일 | 상태 | 로컬 최소 이득 vs v21 |
|---|---|---:|
| `candidates/submit_v22_conservative.zip` | 사전 등록 분기에 따라 제출 중단 | +3.9484 |

v22의 Public 변화가 `0 ~ +3` 구간이므로 보수형과 단순 강도 조정은 제출하지 않는다. 상세 근거는 [`../reports/v22_public_result_20260817.md`](../reports/v22_public_result_20260817.md)에 있다.

## 폴더 역할

```text
프로젝트 루트/
├─ submit_v26.zip          현재 champion, 제출·검증 기준 파일
├─ submit_v25.zip          v26 직접 부모
├─ submit_v22.zip          v25 직접 부모
├─ submit_v21.zip          v22 직접 부모
└─ submissions/
   ├─ README.md            이 안내서
   └─ history/             과거 제출본과 미채택 후보 ZIP
```

`history/`의 ZIP은 점수 계보와 재패키징을 위해 보존한다. 새 제출 후보를 만들 때는 루트에 생성하고, Public 결과가 확정된 뒤 champion이 아니면 `history/`로 옮긴다.

v26의 직접 부모는 루트의 `submit_v25.zip`이고, v25의 직접 부모는 `submit_v22.zip`이다. v27과 v28은 실제 제출했으나 champion(v26)을 넘지 못한 probe로 보존한다.

## 주요 Public 이력

| 파일 | Public 점수 | 상태 |
|---|---:|---|
| `../submit_v26.zip` | **1157.9736407889** | 현재 champion (제출 ID `51773`) |
| `../submit_v27.zip` | 1156.6153781694 | v25 기반 10% probe, champion 미달 |
| `../submit_v28.zip` | 1156.9983034655 | v27 기반 손잡이 routing, champion 미달 |
| `../submit_v25.zip` | 1155.8293405409 | v26 직접 부모 |
| `../submit_v22.zip` | 1153.0436023798 | v25 직접 부모 |
| `../submit_v21.zip` | 1151.5138356157 | v22 직접 부모 |
| `history/submit_v20.zip` | 1151.472428719 | v21 직접 부모 |
| `history/submit_v19.zip` | 1144.1518063753 | v20 직접 부모 |
| `history/submit_v17.zip` | 1093.3213473808 | v19 직접 부모 |
| `history/submit_v18.zip` | 1090.4672420401 | 공격적 F 확장 기각 |
| `history/submit_v13_fixed.zip` | 1068.4365711741 | exact-ASOF 역사 기준선 |
| `history/submit_v2.zip` | 763.2665303697 | 초기 hybrid 기준선 |
| `history/submit.zip` | 749.5249490965 | 최초 기준선 |

전체 제출 ID, 해시, 가설과 판정은 [`../reports/submissions.csv`](../reports/submissions.csv)에 있다. `submit_gt_v1.zip`은 `submit_candidate_game_type_regime_v1_clean.zip`과 SHA-256이 완전히 같아 중복본만 제거했다.
