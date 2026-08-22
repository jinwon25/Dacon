# 제출 파일 보관 규칙

현재 DACON Public 챔피언은 `submit_v104_probe.zip`으로 제출한 **1162.6302840289**다.
전달 기준은 내용과 SHA-256이 동일한
[`../artifacts/standalone_champion_1162/standalone_champion_1162.zip`](../artifacts/standalone_champion_1162/standalone_champion_1162.zip)이다.

| 항목 | 값 |
|---|---:|
| Public 점수 | **1162.6302840289** |
| 확인 당시 순위 | **13위** |
| 제출 ID | `60626` |
| 단일 릴리스 SHA-256 | `0C3B6A9D88D31D9AC32FB43FFD642FA07BFCC3367699FA71841E35188FBDA0BA` |
| 직전 1161 대비 | **+1.4282239867** |
| 목표 1170까지 | **7.3697159711** |
| 목표 1200까지 | **37.3697159711** |

## 최신 승격 결과

| 파일 | 상태 | OOF gain full/late-2024 | SHA-256 |
|---|---|---:|---|
| `../artifacts/standalone_champion_1162/standalone_champion_1162.zip` | API 성공, Public 1162.6302840289 | +3.2162 / +2.5868 | `0C3B6A9D…FBDA0BA` |
| `../artifacts/standalone_champion_1161/standalone_champion_1161.zip` | API 성공, Public 1161.2020600422 | +1.0181 / +2.9166 | `C033FC38…FE6C4F7` |

v104는 v84 위의 안정 R_CORE 행에 독립 FM과 paired conditional correction을 적용한
단일 실행 패키지다. Public gain `+1.4282239867`을 확인해 챔피언으로 승격했다. 상세
근거는 [`../reports/target1170_v104_public_result_20260822.md`](../reports/target1170_v104_public_result_20260822.md)에 있다.

## 제출하지 않을 동일 계열 후보

| 파일 | 상태 | 로컬 최소 이득 vs v21 |
|---|---|---:|
| `candidates/submit_v22_conservative.zip` | 사전 등록 분기에 따라 제출 중단 | +3.9484 |

v22의 Public 변화가 `0 ~ +3` 구간이므로 보수형과 단순 강도 조정은 제출하지 않는다. 상세 근거는 [`../reports/v22_public_result_20260817.md`](../reports/v22_public_result_20260817.md)에 있다.

## 폴더 역할

```text
프로젝트 루트/
├─ artifacts/standalone_champion_1162/
│  ├─ standalone_champion_1162.zip   현재 전달·검증 기준
│  ├─ standalone_manifest.json
├─ artifacts/standalone_champion_1161/  직전 챔피언 감사본
├─ artifacts/standalone_champion_1159/  직전 챔피언 감사본
└─ submissions/
   ├─ README.md            이 안내서
   └─ history/             과거 제출본과 미채택 후보 ZIP
```

`history/`의 ZIP은 점수 계보 감사에만 쓴다. 현재 champion은 이전 ZIP 없이 자체
실행된다. 새 후보는 자동 게이트를 통과한 경우에만 별도 이름으로
생성하고, 현재 champion을 덮어쓰지 않는다.

v26의 직접 부모는 `submit_v25.zip`이고, v25의 직접 부모는 `submit_v22.zip`이다.
v27과 v28은 v26을 넘지 못한 probe다. 이들은 현재 champion의 실행 의존성이
아니며 계보 감사용으로만 보존한다.

## 주요 Public 이력

| 파일 | Public 점수 | 상태 |
|---|---:|---|
| `submit_v84_probe.zip` | **1161.2020600422** | 현재 champion; 제출 ID `59988`; 로컬 전달명 `standalone_champion_1161.zip` |
| `submit_v82_probe.zip` | **1159.3352239501** | v84 직접 부모; 로컬 전달명 `standalone_champion_1159.zip` |
| 팀원 `0819_3_tmgate03.zip` | 1158.0745556751 | 직전 champion; standalone 감사본 보존 |
| `../submit_v26.zip` | **1157.9736407889** | 최종 gate 직접 부모 (제출 ID `51773`) |
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
