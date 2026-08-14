# Public 개선 및 규칙 준수 기록 (2026-08-14)

## 결과

| 제출물 | 핵심 변경 | Public | 순위 | 제출 ID |
|---|---|---:|---:|---:|
| `submit_v2.zip` | 기존 기준선 | 763.2665303697 | - | - |
| `submit_v6.zip` | corrected-state OOF residual, eta 1.0 | 889.4053136645 | 470 | 1531527 |
| `submit_v7.zip` | legacy CatBoost 다양성 축 0.175 | 956.1094025524 | 327 | 1531547 |
| `submit_v8.zip` | 동일 축 가중치 0.35 | 997.7587661493 | 246 | 1531550 |
| `submit_v9.zip` | Public BSS 이차식 최적 가중치 0.5534087426 | 1014.6835364186 | 211 | 1531555 |
| `submit_v10.zip` | compact advanced-domain OOF residual, eta 0.90 | **1015.8076603531** | **212** | 제출 목록 48461 |

- 최초 대비 개선: `+252.5411299834`.
- v9 대비 개선: `+1.1241239345`.
- 1차 목표 1000: 달성.
- 1100 목표: 미달 (현재 대비 `84.1923396469` 필요).
- 2026-08-14 확인 Top 10% 경계(94위): `1079.9157207276`.
- Top 10% 경계까지 격차: `65.2321843090`.
- 최종 확인창에서 잔여 1회를 확인한 뒤 v10에 사용했으며 일일 잔여 횟수는 `0`.
- `submit_v10.zip` SHA-256: `08DCDB49997AE680CCED2F15B561280BAB063D40993825E621AA96024040B345`.

## v10 순방향 OOF 근거

- compact LightGBM 잔차와 기존 legacy CatBoost 축을 결합한 Brier는 O23
  `0.2515292302`, O24 `0.2478814603`이었다.
- O24의 재구성 v9 Brier `0.2479874520` 대비 `-0.0001059917` 개선했다.
- 얕은 15-leaf 구조와 eta `0.90`이 O23·O24에서 같은 방향으로 재현되어 최종 승격했다.
- CatBoost 회귀 잔차는 O24에서 LightGBM보다 Brier `+0.00002948` 열세였다.
- legacy CatBoost 다중 seed 평균은 O24에서는 좋아졌지만 O23에서 악화해 제외했다.
- 선수-플래툰 lookup은 O23·O24 모두 소폭 개선했으나 O24 선수 bootstrap 95% CI가
  `[-0.00002897, +0.00000604]`로 0을 포함해 제외했다.
- 로컬 개선 폭은 Public에서 `+1.1241`점으로만 이어졌다. 따라서 향후에는 2023/2024
  내부 검증 절대 크기보다 Public 분포 전이와 축의 상보성 오차를 더 보수적으로 봐야 한다.

## v10 규칙·배포 감사

- 1행/전체/셔플/2행 청크/타 행 변경/동일 행 5회 반복의 최대 예측 차이는 모두 `0.0`.
- test 내부 `groupby`, `value_counts`, `rolling`, `expanding`, `rank` 및 네트워크 호출이 없다.
- 평가 행은 자신의 입력값과 공식 train으로 고정한 prior·모델만 조회한다.
- ZIP에 test 파일이 없고, 두 residual spec의 train SHA-256이 공식 로컬 train과 일치한다.
- 실제 평가 규모 245,789행 추론은 17.695초, peak RSS 1,429.4MB였다.

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
