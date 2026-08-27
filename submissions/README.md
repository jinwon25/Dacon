# 제출 파일 보관 규칙

마지막 갱신: `2026-08-28 KST`

## 현재 기준

| 항목 | 값 |
|---|---:|
| Public 챔피언 | `submit_jy_runners_high_li_bridge027.zip` |
| 점수 / 순위 | **1172.1373858439** / 제출 이력 기준 확인 |
| 제출 ID | DACON UI 확인값 미기록 |
| SHA-256 | `4C924E046091304BFC73B50BE51110BDF1351DFD9B577738CC6B65A8DFF43C9E` |
| GitHub 전달 경로 | `submissions/releases/jy_runners_high_li_bridge027/submit_jy_runners_high_li_bridge027.zip` |
| v167 대비 / 1170 초과 | **+0.0601478118** / **+2.1373858439** |

최신 JY ZIP은 `script.py`, `requirements.txt`, `model/`만으로 실행되는 172파일 독립 패키지다.
과거 ZIP을 실행 시 참조하지 않는다. 실행·검증 방법은
[`../docs/STANDALONE_CHAMPION.md`](../docs/STANDALONE_CHAMPION.md)를 따른다.

## 최신 JY 제출

`submit_jy_runners_high_li_bridge027.zip`은 v167/main submit을 기준으로 `R_CORE` 중
`num_runners_on > 0` 또는 `li >= 1.5`인 행만 추가 보정한 후보다. 2026-08-28 03:38:21 KST
제출에서 Public `1172.1373858439`를 확인했고 v167 대비 `+0.0601478118`이라 챔피언으로
승격했다. 상세 판단은
[`../reports/jy_runners_high_li_bridge027_public_result_20260828.md`](../reports/jy_runners_high_li_bridge027_public_result_20260828.md)에 있다.

## 주요 Public 계보

| 제출 | Public | 제출 ID | 상태 |
|---|---:|---:|---|
| `submit_jy_runners_high_li_bridge027.zip` | **1172.1373858439** | - | 현재 챔피언 |
| `submit_v167.zip` | 1172.0772380321 | `1547707` | JY 후보 기준선 |
| `submit_v148.zip` | 1170.3014697177 | `1544757` | v167 직접 부모 |
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
├─ submissions/
│  ├─ README.md             이 안내서
│  ├─ releases/v167/
│  │  ├─ submit_v167.zip    PRIVATE 저장소 전달용 확정 파일
│  │  └─ README.md          해시·실행·예외 범위
│  ├─ releases/jy_runners_high_li_bridge027/
│  │  └─ submit_jy_runners_high_li_bridge027.zip
│  └─ history/              과거 제출 계보, ZIP은 Git 제외
├─ artifacts/v167_h1_affine_submission_package_20260825_01/
│  ├─ submit_v167.zip       로컬 빌드 원본, Git 제외
│  └─ manifest.json         빌드·감사 결과, Git 제외
├─ artifacts/v148_v142_v138_blend_package_20260823_01/
│  └─ submit_v148.zip       v167 직접 부모, Git 제외
└─ artifacts/v154_maturity_delta_submission_package_20260823_01/
   └─ submit_v154.zip       과거 비승격 제출, Git 제외
```

새 후보는 기존 챔피언을 덮어쓰지 않고 별도 버전 디렉터리에 만든다. 해시 고정 예외로
명시된 릴리스 외의 로컬 artifact ZIP은 GitHub에 있다고 가정하면 안 된다.

## 보관 원칙

- 공식 데이터, 인증정보, 쿠키·키는 Git/LFS에 올리지 않는다.
- 검증된 v167 ZIP과 최신 JY champion ZIP만 PRIVATE 저장소 전달을 위한 명시적 예외로 추적한다.
- 다른 제출 ZIP·모델·OOF는 문서화된 기존 1161 LFS·1170 OOF allowlist 또는 별도 승인
  없이는 Git/LFS에 올리지 않는다.
- 중복 ZIP은 파일명이 아니라 SHA-256으로 판별한다.
- DACON에 제출한 파일의 해시와 API 결과는 즉시 `reports/submissions.csv`에 기록한다.
- 챔피언 승격은 공식 점수 또는 명시된 검증 근거가 있을 때만 한다.
- v154는 확인된 Public `1168.4038526829`로 기각 상태를 보존한다.
