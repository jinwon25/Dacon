# 제출 파일 보관 규칙

현재 DACON Public 챔피언은 프로젝트 루트의 [`../submit_v20.zip`](../submit_v20.zip)이다.

| 항목 | 값 |
|---|---:|
| Public 점수 | **1151.472428719** |
| 확인 당시 순위 | **12위** |
| 제출 ID | `1534122` |
| SHA-256 | `4E50A6B7C7D970AF9B8A3656C4D87F2A7D856321C53416E831C4B8E5B01BAE29` |
| v19 대비 | **+7.3206223437** |
| 목표 1160까지 | **8.5275712810** |

## 폴더 역할

```text
프로젝트 루트/
├─ submit_v20.zip          현재 champion, 제출·검증 기준 파일
└─ submissions/
   ├─ README.md            이 안내서
   └─ history/             과거 제출본과 미채택 후보 ZIP
```

`history/`의 ZIP은 점수 계보와 재패키징을 위해 보존한다. 새 제출 후보를 만들 때는 루트에 생성하고, Public 결과가 확정된 뒤 champion이 아니면 `history/`로 옮긴다.

v20의 직접 부모는 `history/submit_v19.zip`이다. v20 재패키징·검증은 이 부모와 v20 artifact manifest를 함께 사용한다.

## 주요 Public 이력

| 파일 | Public 점수 | 상태 |
|---|---:|---|
| `../submit_v20.zip` | **1151.472428719** | 현재 champion |
| `history/submit_v19.zip` | 1144.1518063753 | v20 직접 부모 |
| `history/submit_v17.zip` | 1093.3213473808 | v19 직접 부모 |
| `history/submit_v18.zip` | 1090.4672420401 | 공격적 F 확장 기각 |
| `history/submit_v13_fixed.zip` | 1068.4365711741 | exact-ASOF 역사 기준선 |
| `history/submit_v2.zip` | 763.2665303697 | 초기 hybrid 기준선 |
| `history/submit.zip` | 749.5249490965 | 최초 기준선 |

전체 제출 ID, 해시, 가설과 판정은 [`../reports/submissions.csv`](../reports/submissions.csv)에 있다. `submit_gt_v1.zip`은 `submit_candidate_game_type_regime_v1_clean.zip`과 SHA-256이 완전히 같아 중복본만 제거했다.
