# 1150 목표 심층 개선 결과 — joint state / latent failure mode

## 결론

`submit_v17.zip`(Public **1093.3213473808**)을 기준으로 한 사전 `near_1150` 로컬 게이트를 처음 통과한 후보를 만들었다. 신규 후보 **`submit_v19.zip`**은 2026-08-16 13:12:45 KST에 제출되어 Public **1144.1518063753**, 16위를 기록했다. v17 대비 **+50.8304589945** 상승했고 목표 1150까지 **5.8481936247** 남았다.

핵심은 두 신호의 결합이다.

1. **다년도 row-local 상태 모델**: 현재 행의 공식 `asof_*` 값에서 2025 시즌 내 투수·타자 상태를 복원하고, 이전 시즌을 지수 감쇠해 학습한 얕은 LightGBM 잔차를 더한다.
2. **잠재 실패유형 모델**: 학습 범위에서 다음 누적 ASOF snapshot으로 현재 투구의 `bad_location / ball_only / strike_only / other` 라벨을 복원한다. 최종 추론에서는 현재 투구 라벨이나 다른 test 행을 보지 않고, 2019~2024로 학습한 네 클래스 확률만 사용한다.

2024는 반복적으로 본 개발 데이터이므로 이 결과는 독립적인 1150 보장이 아니다. 다만 고정 프로토콜의 모든 수치 문턱, 월별 안정성, 세 가지 군집 bootstrap, 코드 패키지 검증을 모두 통과한 최초의 **1150 근접 제출 후보**다.

## 로컬 근거

| 항목 | 사전 near_1150 문턱 | v19 결과 | 통과 |
|---|---:|---:|---:|
| 최소 forward gain vs v17 | +40 | **+46.396** | 예 |
| 평균 forward gain vs v17 | +50 | **+176.751** | 예 |
| 최신 2024 gain vs v17 | +45 | **+46.396** | 예 |
| 최소 cluster bootstrap p05 | +10 | **+27.110** | 예 |
| 최소 양수 월 비율 | 70% | **100%** | 예 |
| 최소 도메인 gain 허용치 | -10 | **+20.768** | 예 |
| 양수 forward fold | 2 | **2** | 예 |

| 감사 연도 | v17 BSS | v19 BSS | gain |
|---:|---:|---:|---:|
| 2023 | -999.296 | -692.190 | **+307.107** |
| 2024 | 915.870 | 962.265 | **+46.396** |

2024 월별 gain은 3월부터 10월까지 모두 양수였고 최솟값은 10월 **+14.687**였다. 2024 도메인 내부 BSS gain도 `R_CORE +20.768`, `R_ANCHOR +76.244`, `F +155.810`으로 모두 양수였다.

### 군집 bootstrap

5,000회 paired bootstrap의 고정-reference BSS gain 하위 5%:

| 감사 연도 | 투수 | 타자 | 투수×타자 | 개선 표본 비율 |
|---:|---:|---:|---:|---:|
| 2023 | +251.737 | +245.746 | +284.792 | 100% |
| 2024 | **+27.110** | **+28.644** | **+28.917** | 100% |

bootstrap은 개발 과정의 후보 선택 편향을 제거하지 못한다. 따라서 Public 제출은 배포 probe로 해석해야 한다.

## 선택된 도메인 라우팅

| 도메인 | 상태 신호 | 실패유형 신호 | 두 폴드 최소 기여 |
|---|---|---|---:|
| R_CORE | h=0.5, 투수 가중 0.75/1.00 pair; 보정 `abs<.02`, v17 `.4~.6`, 모델 4/6 방향 합의 | 세 반감기 mode 확률 평균 × mode별 제구 출력 | +14.630 |
| R_ANCHOR | h=0.5, 투수 가중 0.75/0.50 pair; 양의 `.02~.04` 상태 보정 | h=0.5 mode 확률의 제곱근 온도 × mode 조건부 성공률 | +13.430 |
| F | global/domain h=0.5 pair; `abs<.04`, 모델 5/6 방향 합의 | h=0.5 mode 확률의 1.5승 온도 × mode 조건부 성공률 | +18.337 |

## 패키지 검증

| 항목 | 결과 |
|---|---:|
| 후보 | `submit_v19.zip` |
| parent | `submit_v17.zip` |
| ZIP SHA-256 | `B12070D8017AE6A78BFC056F392BACDA931F8ED7439AA9821880FC986CE962B2` |
| 크기 | 30.789 MB |
| 공식 5행 smoke | 통과, 4.80초, peak 272 MB |
| 대표 245,789행 | 통과, **58.05초**, peak **1.423 GB** |
| 예측 범위 | 0.356162 ~ 0.653832 |
| 전체/분할 배치 최대 차이 | `1.11e-16` |
| 네트워크 사용 | 없음 |
| test 집계/순서 사용 | 없음 |
| 세 도메인 변경 | 모두 확인 |

검증 상세는 `reports/v19_validation.json`, 로컬 후보 상세는 `artifacts/state_mode_joint_20260816_02/summary.json`, 최종 학습 hash는 `artifacts/state_mode_joint_final_20260816/manifest.json`에 있다.

## 커뮤니티·공식 FAQ 확인

2026-08-16 재검색 기준, 공개 Talk/코드공유/일반 웹에서 상위권 참가자의 구체적인 점수 개선 레시피나 공개 솔루션은 찾지 못했다. 코드공유에는 공식 RandomForest 베이스라인 계열만 확인됐다. 따라서 참가자 팁을 모방하기보다 공식 FAQ가 허용한 정보 범위를 모델 설계에 반영했다.

- [대회 FAQ](https://dacon.io/competitions/official/236743/talkboard/417082): 과거 TrackMan을 이용한 ID 추정·프로파일, 학습 범위의 현재 투구 TrackMan을 teacher/soft label로 쓰는 것, 학습 범위 안의 미래 행을 이용한 학습 라벨 보강이 허용된다는 답변을 확인했다.
- [평가 행 독립 추론 공지](https://dacon.io/competitions/official/236743/talkboard/417123): 다른 test 행의 분포·빈도·순서를 사용하지 않아야 한다. v19는 각 행의 공식 ASOF와 동결된 2019~2024 아티팩트만 사용한다.
- [주요 규칙 공지](https://dacon.io/competitions/official/236743/talkboard/417094) 및 [대회 규칙](https://dacon.io/competitions/official/236743/overview/rules): 외부 데이터 금지, 코드 재현성, 누수 금지를 지켰다.
- [코드공유](https://dacon.io/competitions/official/236743/codeshare): 공개 참가자 고득점 해법은 확인되지 않았다.

## 참고 문헌과 반영 여부

- [Generalized Distillation](https://arxiv.org/abs/1511.03643), [Privileged Feature Distillation](https://arxiv.org/abs/1907.05171), [Privileged-feature caution](https://arxiv.org/abs/2209.08754): 현재 투구 사후 정보를 teacher로만 쓰고 inference-safe student로 압축하는 실험 설계에 반영했다. 단순 TrackMan PFD는 2024에서 재현되지 않아 최종 후보에서 제외했다.
- [TabM, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html), [공식 구현](https://github.com/yandex-research/tabm): 표형 앙상블 대안으로 검토했지만, 현재 성공 후보가 얕은 모델과 해석 가능한 잠재모드로 게이트를 통과했고 CPU/패키지 비용을 고려해 이번 v19에는 포함하지 않았다.
- [Release parameters and pitch location](https://pmc.ncbi.nlm.nih.gov/articles/PMC7739723/), [xCTRL](https://arxiv.org/abs/2508.19184): 투수·구종·타자 손별 제구 분포가 다르다는 야구 도메인 근거로 활용했다. 실제 구종 oracle은 강했지만 독립 행 구종 예측이 병목이어서 최종 주축은 실패유형 잠재변수가 됐다.

## 주요 기각 실험

- 현재 투구 TrackMan oracle은 컸지만 표준 PFD student는 2024에서 약 0 또는 음수였다.
- 과거 TrackMan 물리 프로파일 직접 입력은 2023/2024 모두 약 -20점이었다.
- actual 투수×타자 matchup EB, 고정 context 잔차, 전역 calibration은 0~수 점 또는 음수였다.
- 구종 잠재혼합은 실제 구종 oracle이 2024 +201.5였으나 예측 가능한 구종 확률의 최대 이득은 +9.52에 그쳤다.
- 다년도 상태 모델 단독은 최신 +27.15, 실패유형 단독은 +20.53이었고, 공동 도메인 라우팅에서만 +46.40에 도달했다.

## 운영 판단

`submit_v19.zip`은 로컬 근거와 코드 검증 기준으로 다음 단일 Public probe에 사용할 수 있다. 다만 Public 1150을 보장하지 않으며, 제출 후 점수에 맞춰 같은 계열의 threshold/온도/가중치를 미세조정하면 리더보드 과적합이 된다. 제출을 실행할 경우 v19 한 번으로 가설을 검증하고, 결과·runtime·submission ID·ZIP hash를 `reports/submissions.csv`에 즉시 기록한다.
