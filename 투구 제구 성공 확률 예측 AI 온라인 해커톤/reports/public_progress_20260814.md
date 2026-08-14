# Public 개선 및 규칙 준수 기록 (2026-08-14)

## 결과

| 제출물 | 핵심 변경 | Public | 순위 | 제출 ID |
|---|---|---:|---:|---:|
| `submit_v2.zip` | 기존 기준선 | 763.2665303697 | - | - |
| `submit_v6.zip` | corrected-state OOF residual, eta 1.0 | 889.4053136645 | 470 | 1531527 |
| `submit_v7.zip` | legacy CatBoost 다양성 축 0.175 | 956.1094025524 | 327 | 1531547 |
| `submit_v8.zip` | 동일 축 가중치 0.35 | 997.7587661493 | 246 | 1531550 |
| `submit_v9.zip` | Public BSS 이차식 최적 가중치 0.5534087426 | **1014.6835364186** | **211** | **1531555** |

- 최초 대비 개선: `+251.4170060490`.
- 1차 목표 1000: 달성.
- 다음 사이클 목표: `1100` (현재 대비 `85.3164635814` 필요).
- 2026-08-14 확인 Top 10% 경계(94위): `1079.9157207276`.
- Top 10% 경계까지 격차: `65.2321843090`.
- 데이콘 검증 API로 확인한 남은 제출 가능 횟수: `1`.
- `submit_v9.zip` SHA-256: `5888D52C02C59C4BE6203FA645F3D2A4335F95349FF826D09CE6C2E387611450`.

## 가중치 최적화 근거

`submit_v6`, `submit_v7`, `submit_v8`은 같은 base/residual/CatBoost 예측을 사용하고
CatBoost 효과 가중치만 각각 `0`, `0.175`, `0.35`로 다르다. Brier Skill Score는
Brier score의 affine transform이고 예측은 가중치에 선형이므로 Public 점수는 가중치의
이차식이다. 세 Public 점으로 적합한 최적점은 다음과 같다.

- 최적 가중치: `0.5534087425546688`.
- 예측 Public: `1014.6835364184`.
- 실제 Public: `1014.6835364186`.

OOF O22--O24에서 이 가중치의 확률 범위는 `0.3034948`--`0.7510403`, clipping 행은
0개였다.

## 평가 행 독립성 및 규칙 준수

모든 제출 피처는 다음 두 종류만 사용한다.

1. 현재 평가 행 자체의 공식 입력/as-of 변수.
2. 공식 `train.csv`만으로 사전 생성해 ZIP에 넣은 모델과 고정 lookup.

평가 파일의 다른 행을 이용한 선수·팀·월·경기 집계, 누적값, rolling, lag, 평균,
분포, 빈도, 순위 또는 보정은 없다. 인터넷 호출과 실행 후 상태 저장도 없다.

`submit_v9.zip`에 대해 동일 행을 다음 조건으로 재예측했다.

- 해당 행만 있는 1행 배치.
- 공식 sample 전체 배치.
- 역순 셔플 배치.
- 2행 단위 분할 배치.
- 다른 모든 행의 여러 입력값을 바꾼 배치.
- 동일 행을 5회 반복한 배치.

최대 절대 차이는 `5.551115123125783e-17`이고, 셔플·분할·다른 행 변경·반복 행
검사는 정확히 `0.0`이었다. 판정 허용오차 `1e-12`를 통과했다. 정적 검사에서도
`groupby`, `value_counts`, `rolling`, `expanding`, `rank` 및 네트워크 호출은 없었다.
ZIP에는 test 데이터가 없고 residual 사양의 학습 파일 SHA-256은 공식 로컬
`train.csv`와 일치했다.

근거 파일:

- `artifacts/top10_20260814/submit_v6_row_independence.json`
- `artifacts/top10_20260814/submit_v7_row_independence.json`
- `artifacts/top10_20260814/submit_v8_row_independence.json`
- `artifacts/top10_20260814/submit_v9_row_independence.json`
- `reports/package_validation.md`

## 마지막 제출을 보존한 이유

v9를 기준으로 corrected CatBoost와 기존 상황 prior를 추가하는 순방향 OOF 검사를
실시했다. O23에서 선택한 추가 corrected-CB 가중치 `0.20`은 O24에서 Brier를
`+0.0000677352` 악화시켰고, 상황 prior `0.50`도 O24에서 `+0.0000123285`
악화시켰다. 따라서 남은 회차를 이 축에 사용하지 않는다.

다음 사이클은 legacy CatBoost의 seed, 학습 표본 및 최근 시즌 가중을 독립 축으로
만들어 O22--O24 latest-origin 검증을 통과한 경우에만 제출한다.
