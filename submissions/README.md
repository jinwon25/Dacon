# 제출 파일 보관 규칙

마지막 갱신: `2026-08-23 KST`

## 현재 기준

| 항목 | 값 |
|---|---:|
| Public 챔피언 | `submit_v148.zip` |
| 점수 / 순위 | **1170.3014697177** / **9위** |
| 제출 ID | `1544757` |
| SHA-256 | `7A27BE5878A79934544C741F283C139D40FB20484D52DB494928BCBE27E1E337` |
| 로컬 전달 경로 | `artifacts/v148_v142_v138_blend_package_20260823_01/submit_v148.zip` |
| v142 대비 / 1170 초과 | **+0.6736764355** / **+0.3014697177** |

v148 ZIP은 `script.py`, `requirements.txt`, `model/`만으로 실행되는 89파일 독립 패키지다.
과거 ZIP을 실행 시 참조하지 않는다. 실행·검증 방법은
[`../docs/STANDALONE_CHAMPION.md`](../docs/STANDALONE_CHAMPION.md)를 따른다.

## 마지막 v154 제출

`submit_v154.zip`은 85% bridge에 outcome-free 5월 maturity delta distillation을 추가한
마지막 일일 제출이다. 2026-08-23 15:09:43 KST API 접수는 성공했고 누적 제출은 35회가
됐다. 이후 best-only 리더보드는 v148의 점수·ID·9위를 유지했다.

로그인 세션 없이는 v154의 개별 점수나 실패 사유를 확인할 수 없으므로 실패로 단정하지
않는다. 다만 챔피언 승격 증거가 없어 v148을 계속 전달한다. v154의 SHA-256은
`979904DB258A15DDEC65024359DB152269B103567D59C93F75E57DA3E39A5C42`다. 상세 판단은
[`../reports/target1173_v149_v154_final_submission_20260823.md`](../reports/target1173_v149_v154_final_submission_20260823.md)에 있다.

## 주요 Public 계보

| 제출 | Public | 제출 ID | 상태 |
|---|---:|---:|---|
| `submit_v148.zip` | **1170.3014697177** | `1544757` | 현재 챔피언 |
| `submit_v142.zip` | 1169.6277932822 | `1544738` | v148 직접 부모 |
| `submit_v124.zip` | 1164.2949203402 | `1544631` | v142 직접 부모 |
| `submit_v116.zip` | 1161.5978783152 | `61030` | 교차-origin failure prior, 기각 |
| `submit_v104_probe.zip` | 1162.6302840289 | `60626` | v124 계보 부모 |
| `submit_v84_probe.zip` | 1161.2020600422 | `59988` | F shared FM 부모 |
| `submit_v82_probe.zip` | 1159.3352239501 | - | R_CORE strict 부모 |
| `0819_3_tmgate03.zip` | 1158.0745556751 | - | TrackMan-ASOF 부모 |
| `submit_v26.zip` | 1157.9736407889 | `51773` | R_ANCHOR 15% 부모 |
| `submit_v27.zip` | 1156.6153781694 | `51775` | v26 미달 |
| `submit_v28.zip` | 1156.9983034655 | `51783` | v26 미달 |
| `submit_v25.zip` | 1155.8293405409 | `1535141` | v26 직접 부모 |
| `submit_v22.zip` | 1153.0436023798 | - | v25 직접 부모 |
| `submit_v21.zip` | 1151.5138356157 | `1534213` | v22 직접 부모 |
| `submit_v20.zip` | 1151.4724287190 | - | v21 직접 부모 |
| `submit_v19.zip` | 1144.1518063753 | - | v20 직접 부모 |
| `submit_v17.zip` | 1093.3213473808 | - | 압박 잔차 부모 |
| `submit_v13_fixed.zip` | 1068.4365711741 | `1532352` | exact-ASOF 기준선 |

공식 제출 ID, 전체 해시, 가설, 실행시간과 판정의 단일 원장은
[`../reports/submissions.csv`](../reports/submissions.csv)다. 과거 문서의 “현재 champion”은
작성 당시 기록으로만 해석한다.

## 폴더 역할

```text
프로젝트 루트/
├─ artifacts/v148_v142_v138_blend_package_20260823_01/
│  ├─ submit_v148.zip       현재 로컬 전달 파일, Git 제외
│  └─ manifest.json         빌드·감사 결과, Git 제외
├─ artifacts/v154_maturity_delta_submission_package_20260823_01/
│  └─ submit_v154.zip       마지막 비승격 제출, Git 제외
└─ submissions/
   ├─ README.md             이 안내서
   └─ history/              과거 제출 계보, ZIP은 Git 제외
```

새 후보는 기존 챔피언을 덮어쓰지 않고 별도 버전 디렉터리에 만든다. 로컬 artifact 경로는
코드와 문서가 참조할 수 있지만, ZIP 자체가 GitHub에 있다고 가정하면 안 된다.

## 보관 원칙

- 공식 데이터, 인증정보, 쿠키·키는 Git/LFS에 올리지 않는다.
- 대용량 제출 ZIP과 모델은 승인된 비공개 팀 채널에서 SHA-256과 함께 전달한다.
- 중복 ZIP은 파일명이 아니라 SHA-256으로 판별한다.
- DACON에 제출한 파일의 해시와 API 결과는 즉시 `reports/submissions.csv`에 기록한다.
- 챔피언 승격은 공식 점수 또는 명시된 검증 근거가 있을 때만 한다.
- 개별 점수가 확인되지 않은 v154는 성공·실패를 추정하지 않고 비승격 상태로 보존한다.
