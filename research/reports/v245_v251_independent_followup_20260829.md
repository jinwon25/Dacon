# v244 Public 승격 및 v245–v251 독립 후속 연구

작성 시각: 2026-08-29 KST

## 결론

기존 팀 챔피언 `bridge027_active50_w030.zip`의 정확한 Public 점수는
`1175.2737119427`이고, v244의 Public 점수는 `1175.9746833121`이다. 따라서 v244는
`+0.7009713694점` 개선된 새 챔피언이다. 목표 `1180`까지는 `4.0253166879점` 남았다.

v244 승격 뒤 독립 모델 계보 세 가지를 추가로 감사했다.

1. swimmer 저차원 Logistic/RF/HistGB OOF
2. v244와 동일한 114개 runtime-faithful 피처의 독립 LightGBM
3. XGB–LightGBM rolling meta stack 및 보호 구간 R_ANCHOR specialist

어느 후보도 사전 정의한 source 동시 양수와 locked/cluster-bootstrap 조건을 모두 통과하지
못했다. 따라서 새 제출 ZIP은 만들지 않았고, v244 재현 ZIP과 기존 Public1175 릴리스는
수정하지 않았다.

## 1. 정확한 Public 비교

| 제출 | ID | Public | 실행 시간 | 판정 |
|---|---:|---:|---:|---|
| `bridge027_active50_w030.zip` | 72710 | 1175.2737119427 | 110초 | 이전 챔피언 |
| `submit_v244.zip` | 73881 | 1175.9746833121 | 110초 | 새 챔피언 |

v244의 사전 중심 예측 `1178.3227258660`은 실제보다 `2.3480425539점` 높았다. 기존
fallback 계열의 로컬→Public 절대 개선폭 환산식은 폐기하고, 실제 `+0.7009713694점`을
다음 가중치 선택에 재사용하지 않았다.

## 2. v245–v246: swimmer 독립 OOF

공개 팀 브랜치의 원본 코드를 수정하지 않고 공식 train만으로 2022/2023/2024
strict-forward OOF를 재생성했다. Logistic, RandomForest, HistGradientBoosting 및 사전
고정 RF–HistGB 평균을 v244에 `0.25%–5%` 혼합했다.

최소 비중 `0.25%`의 source gain도 다음과 같았다.

| 모델 | full-2022 | late-2023 | full-2024 |
|---|---:|---:|---:|
| Logistic | +0.323849 | -0.164141 | -0.001570 |
| RandomForest | +0.189434 | -0.173085 | -0.177884 |
| HistGB | +0.419974 | -1.641561 | -0.127533 |
| RF–HistGB 평균 | +0.305035 | -0.907019 | -0.152353 |

두 source 축 동시 양수 후보가 없어 `source_reject`로 종료했다.

## 3. v247–v249: runtime-faithful LightGBM

XGB와 같은 114개 피처, 같은 frozen lookup, 같은 prior-season sample weight를 쓰되 학습기만
LightGBM으로 바꿨다.

| 연도 | XGB BSS | LightGBM BSS |
|---|---:|---:|
| 2022 | 482.2314 | 434.4879 |
| 2023 | 426.6679 | 441.0556 |
| 2024 | 600.9391 | 622.9943 |

LightGBM은 2023·2024 단독 성능이 XGB보다 높았지만, v244의 route 안에서 XGB 확률을
부분 교체하면 source gain이 동시에 양수가 아니었다. 예를 들어 LightGBM 25% 혼합은
`-0.063102 / -0.474372 / +0.320329`였다. 모델 근접도·같은 방향·보수성으로 제한한
6개 consensus scope도 모두 source에서 기각됐다.

v244 재구성 최대 절대 오차는 세 축 모두 `1.11e-16`이므로 이 기각은 부모 또는 route
재현 오류 때문이 아니다.

## 4. v250: strict-forward rolling meta stack

2021 XGB/LightGBM OOF를 추가 생성했다. audit 연도별 meta LogisticRegression은 오직 이전
연도 OOF와 라벨로만 학습했다.

- 2022 meta: 2021만 학습
- 2023 meta: 2021–2022 학습
- 2024 meta: 2021–2023 학습

v244 대비 gain은 `full-2022 +19.5474`, `late-2023 -3.3107`, `full-2024 +1.2421`이었다.
late-2023 source가 음수였고 2024 pitcher/crossed/block bootstrap p05도 각각
`-5.5294 / -10.4574 / -5.9127`, Reality Check `p=.2594`라 기각했다.

## 5. v251: R_ANCHOR specialist

v244가 수정하지 않는 팀 13 관련 Regular 행만 별도 route로 두고 XGB, LightGBM,
두 모델 평균을 `5%–30%` 혼합했다. 활성 행은 2022/late-2023/2024에서
`42,806 / 16,577 / 44,768`개였다.

가장 작은 5% 혼합도 다음과 같이 late-2023에서 크게 반전됐다.

| 모델 | full-2022 | late-2023 | full-2024 |
|---|---:|---:|---:|
| XGB | +3.1182 | -7.9548 | +0.6952 |
| LightGBM | +1.8630 | -7.5321 | +1.1168 |
| 평균 | +2.5010 | -7.7378 | +0.9178 |

R_ANCHOR fallback 확장은 `source_reject`로 종료했다.

## 6. 보존 및 다음 연구 경계

- v244를 새 Public 기준선으로 사용한다.
- v245–v251은 연구 artifact만 보존하고 제출 후보로 승격하지 않는다.
- XGB/LightGBM 고정 혼합, consensus scope, rolling stack, R_ANCHOR 확장은 같은 결과를
  Public weight 조정으로 재탐색하지 않는다.
- 다음 후보는 이 fallback 확장 계보가 아닌 독립 feature 계보여야 한다.
- 전체 테스트: `565 passed, 21 skipped, 1 existing warning`.

현재 신뢰 가능한 점수 예측은 v244의 실측 `1175.9746833121`뿐이다. 통과 후보가 없는
상태에서 `1180+`를 예측하는 것은 근거가 없으므로 별도 ZIP을 만들지 않는다.
