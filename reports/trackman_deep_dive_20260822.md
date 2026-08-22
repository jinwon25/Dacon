# TrackMan 데이터 전수조사와 1170 경로 재설계 — 2026-08-22

## 결론

현재 Public `1158.0745556751` champion은 유지한다. TrackMan 원자료를 현재 test 투구의
직접 feature로 넣는 경로는 데이터 구조상 불가능하고, 과거 시즌의 투수별 물리 프로필을
origin 전에 동결해 행별 ID에 붙이는 경로만 배포 가능하다. 이 경로를 127개 후보까지
조사했지만 고차원 물리 모델, 변동성 모델, 투구 유형 학생 모델은 모두 시간축에서
무너졌다.

2024 재사용 축에서 유일하게 반복 양수였던 후보는 투수의 과거 평균 induced vertical
break(IVB) 한 개를 4~9월에만 쓰는 v70이었다. 이후 recipe를 그대로 고정한 v74 과거축
검증에서 2022 source OOF가 보정 eta를 두 번 모두 `0`으로 선택했다. v70은 최근 시즌
특화 사후 가설로 최종 기각하며 Public probe나 standalone 패키지를 만들지 않는다.

## 1. 제공 데이터 전수조사

| 항목 | 확인 결과 |
|---|---:|
| 공식 train | 1,475,092행 × 49열, 2019~2024 |
| TrackMan | 1,793,078행 × 30열, 2019~2024 |
| TrackMan 경기 / 투수 | 5,980 / 906 |
| target을 쓰지 않은 구조 정렬 | 1,217,598행, train의 82.5439% |
| 경기별 정렬 Dice 최소 / 중앙값 | 0.9500 / 1.0000 |
| 정렬 margin 최소 / 중앙값 | 0.9279 / 0.9839 |
| 원시 물리량과 target의 최대 절대상관 | 0.08355 |
| season×pitcher×pitch family 내부 최대 절대상관 | 0.03656 |
| 투수 물리 프로필 연도 간 상관 중앙값 / 최소 | 0.90872 / 0.79984 |

정렬은 경기의 투구 이벤트 signature만 사용하며 target 열은 읽지 않는다. 양쪽 row index는
모두 유일했고 경기 수준 일치도와 차순위 margin도 높았다. 따라서 정렬은 TrackMan ID와
공식 ID 사이의 과거 연결 및 데이터 품질 감사에는 충분하다. 그러나 test에는 현재 투구의
TrackMan ID나 물리량이 없으므로, 정렬 자체가 현재 투구 물리량을 복구해 주지는 않는다.

### 제공되지 않는 결정적 변수

- 실제 plate location과 포수의 intended target이 없다.
- test 행에는 현재 투구의 구속·회전·무브먼트·릴리스 정보가 없다.
- 따라서 현재 투구의 제구 오차를 물리 궤적으로 직접 계산할 수 없다.
- 과거 투수 프로필은 “어떤 공을 던지는가”를 나타내지만, “이번 공을 어디로 의도했고
  얼마나 빗나갔는가”를 직접 나타내지 않는다.

이 구분이 병목의 핵심이다. 공개된 pitch-location 연구는 릴리스 위치·속도·회전·투구
문맥을 현재 투구 단위로 사용하지만, 본 대회의 추론 열에서는 그 정보를 쓸 수 없다.

## 2. ID 연결 방법과 배포 범위

target-free 정렬로부터 `support >= 20`, `purity >= 0.99`인 직접 다수결 map을 만들었다.

| origin | 수용 ID | query 행 coverage | query 투수 coverage |
|---:|---:|---:|---:|
| 2021 | 419 | 78.51% | 71.50% |
| 2022 | 510 | 82.54% | 73.33% |
| 2023 | 574 | 84.38% | 78.27% |
| 2024 | 639 | 79.21% | 75.19% |
| 2025(2024 proxy) | 717 | 99.05% | 95.14% |

직접 map과 기존 pitch-mix Hungarian map은 겹치는 443개 ID에서 58.92%만 같은 TrackMan
투수를 골랐다. 직접 map은 event alignment의 강한 일대일 근거를 쓰므로 ID 감사에는 더
적합하지만, 기존 champion gate를 그대로 교체했을 때는 late-2024가 악화됐다. 즉 map의
정확도와 최종 예측 성능은 별개의 문제다.

배포 가능한 TrackMan feature의 시간 계약은 다음과 같다.

```text
행의 origin = Y
사용 가능한 TrackMan 원자료 = season < Y
사전 계산 단위 = 공식 pitcher_id별 동결 profile
추론 입력 = 현재 행의 pitcher_id + 현재 행의 공식 사전투구 변수
금지 = 다른 test 행, test 집계, 현재 투구 TrackMan 물리량의 가정/복원
```

## 3. 물리 feature 조사

v67은 투수 전체 및 fastball/breaking/offspeed별 평균·표준편차, within-pitch-family
표준편차, 최신 시즌 값과 장기 값의 차이 등 127개 profile feature를 만들었다. 모두
`TrackMan season < origin`으로 고정했다.

coverage는 late-2023 81.99%, early-2024 79.64%, late-2024 78.23%였다. 세 시간축에서
상관 부호가 같고 최소 절대상관이 0.05 이상인 feature×domain 조합은 23개였다. 가장 큰
관계는 R_ANCHOR의 장기 평균 IVB 잔차로 `-0.1405 / -0.1517 / -0.1923`이었다.

그러나 이 값들은 투수 단위 반복 관측이 섞인 탐색 상관이며 인과 효과가 아니다. 127개
feature와 여러 domain을 동시에 본 다중 탐색이므로, 상관만으로 모델을 채택하지 않고
시간 OOF에서 incumbent 대비 Brier gain을 재검증했다.

## 4. 1158 위 TrackMan 실험 결과

| 버전 | 가설 | late-2023 | full-2024 | late-2024 | 판정 |
|---|---|---:|---:|---:|---|
| v66 | 직접 ID map으로 기존 gate 교체 | +0.0687 | -0.0155 | -0.1527 | 기각 |
| v68 | R_ANCHOR 저용량 profile Ridge | IVB 0 | IVB 0 | IVB +1.933 | 불안정·기각 |
| v69 | profile Ridge domain 확대 | IVB +1.729 | IVB +1.852 | 포함 | 월 최악 -12.112, 기각 |
| v70 | IVB 1개, ALL, 4~9월 | +0.360¹ | +0.360¹ | +2.805² | v74에서 기각 |
| v71 | 기존 latent pitch type shift 재결합 | +26.010 | -1.420 | -1.060 | 기각 |
| v73 | TrackMan label로 pitch type 학생 재학습 | -0.924 | -2.778 | -1.749 | 기각 |
| v74 | 고정 v70의 2022 과거축 반증 | 0³ | 0⁴ | 확인축 +2.777 | **v70 기각** |

¹ late-2023를 source로 학습하고 full-2024를 감사한 전이 결과다. ² early-2024를 source로
학습하고 late-2024를 감사한 전이 결과다. 표의 v70 열은 서로 다른 두 시간 전이를
명시한 것이며 동일 모델의 세 분할값으로 오해하면 안 된다. ³ early-2022→late-2022,
⁴ full-2022→full-2023이며 두 축 모두 source-only eta가 `0`이었다.

고차원 physics/repeatability feature는 투수 ID와 기존 ASOF prior가 이미 담고 있는 정보를
중복하거나 시즌별 관계 변화에 과적합했다. TrackMan pitch type label로 학생 classifier의
정확도를 약 1%p 높여도 최종 outcome 예측은 악화됐다. 병목이 단순 투구 유형 label 오류가
아니라, 위치 의도 부재와 물리량→제구성공 간 transfer에 있음을 보여 준다.

### v70/v74의 정확한 상태

v70은 한 feature만 사용하고 Ridge alpha와 blend eta를 source OOF에서만 정한다.

- late-2023 → full-2024: `+0.3602`, 4~9월 6개월 모두 양수, 최악 월 `+0.1652`,
  최소 domain `+0.2442`
- early-2024 → late-2024: `+2.8048`, 8~9월 모두 양수, 최악 월 `+2.7063`,
  최소 domain `+0.8319`
- 평균 절대 보정량: 각각 `0.000099`, `0.001282`
- 현재 행의 ID만 사용하고 다른 test 행·test 분포는 사용하지 않는다.

통계 모양은 가장 좋지만 3월·10월 손실을 본 뒤 4~9월 gate를 정의했다. 이를 반증하기
위해 v74는 feature·domain·월·source-only 선택 규칙을 모두 고정한 뒤 과거축을 열었다.

- early-2022 → late-2022: source eta `0`, gain `0`
- full-2022 → full-2023: source eta `0`, gain `0`
- full-2023 → full-2024 확인축: `+2.7774`이나 활성 월 비율 66.67%, 최악 월
  `-3.8585`, 최소 domain `-1.0408`

오래된 두 primary axis가 보정 자체를 선택하지 않았으므로
`passes_independent_older_axes=false`, `decision=reject_v70`이다. v70/v74를 제출 ZIP으로
만들거나 champion pointer를 바꾸지 않는다.

## 5. TrackMan pitch type label 감사

TrackMan 3분류 구종과 기존 ASOF 재구성 label은 1,202,669개 공통 행에서 93.3251%
일치했다. TrackMan만 추가로 3분류 label을 주는 행은 1,520개였다. 기존 classifier의
정확도는 2023에서 TrackMan 기준 51.68%, 재구성 기준 55.61%; 2024에서 각각 51.86%,
56.49%였다.

v73 재학습은 TrackMan 기준 정확도를 2023 52.60%, 2024 52.95%로 올렸지만 최종 Brier
gain은 모든 주요 시간축에서 음수였다. 이후에는 pitch type 학생 구조나 half-life/weight를
미세 조정하지 않는다.

## 6. 문헌·공개 자료와 본 데이터의 연결

- [TrackMan 공식 측정 항목](https://www.trackman.com/baseball/Portable-B1/what-we-track)은
  release speed, spin, movement, release position, plate location 등을 현재 투구 단위로
  추적한다. 본 파일은 그중 일부 물리량만 과거 보조 데이터로 제공한다.
- [Sports Engineering 2025 pitch-location 연구](https://doi.org/10.1007/s12283-025-00497-5)는
  200만 개 이상의 NCAA 투구에서 문맥과 릴리스/궤적 변수를 결합했다. 이런 접근은 현재
  투구 물리량이 없는 본 test에 직접 이식할 수 없고, 과거 profile prior로만 축소 가능하다.
- [릴리스 변수와 투구 위치 연구](https://pubmed.ncbi.nlm.nih.gov/33345028/)와
  [투구 간 ball-flight 변동성 연구](https://pubmed.ncbi.nlm.nih.gov/25809339/)는 제구를
  릴리스·궤적 일관성과 연결하지만, 후자는 시즌 성과와 단순 직접 상관이 없었다. 따라서
  profile 표준편차를 곧바로 “제구력”으로 놓는 것은 과도하다.
- [고속 투구 accuracy 연구](https://pubmed.ncbi.nlm.nih.gov/34376126/) 역시 속도-정확도
  trade-off를 다루지만, 본 대회의 사전투구 열만으로 개별 pitch 결과를 복원해 주지는 않는다.
- [xCTRL](https://arxiv.org/abs/2508.19184)은 위치 분포를 투수×구종×타자손으로 shrinkage
  하지만 실제 위치 자료가 전제다. 여기서는 ASOF prior 설계 원칙만 참고할 수 있다.
- [LUPI 원 논문](https://www.jmlr.org/beta/papers/v16/vapnik15b.html)은 train에만 있는
  privileged information의 원리를 제공한다. v64와 v73 결과상 강한 교사 신호가 없는
  상태에서의 distillation은 우선순위가 낮다.
- 공개 저장소 [jiyunjung0/Baseball](https://github.com/jiyunjung0/Baseball),
  [hoo743-ui/LG_Aimers09](https://github.com/hoo743-ui/LG_Aimers09),
  [x2-qp-cheese/LGAimers](https://github.com/x2-qp-cheese/LGAimers)는 정렬·물리 profile·EDA
  경로를 비교하는 데 참고했다. 공개 보고 수치는 외부 검증 없이 본 모델의 채택 근거로
  사용하지 않았다.

## 7. 규칙 준수와 재현성

[공식 행 독립성 공지](https://dacon.io/competitions/official/236743/talkboard/417123)에 따라
모든 배포 후보는 한 행의 공식 입력, 공식 train, train에서 미리 동결한 artifact만 쓴다.
다른 test 행의 평균·분포·빈도·순서·rolling/lag는 쓰지 않는다. 공식 추가 답변
[417157](https://dacon.io/competitions/official/236743/talkboard/417157)은 Phase 3에서 학습
데이터, 전체 학습·추론 과정, feature와 상수의 도출 근거까지 소명 대상이라고 명시한다.

v65~v73은 test CSV를 읽지 않은 OOF 연구이며, 모든 TrackMan profile은 audit origin보다
이전 시즌에서만 계산했다. 현재 champion 단일 ZIP은 singleton/full/shuffle/partition
행 독립성 검사를 통과했고 과거 ZIP이나 저장소 코드에 의존하지 않는다.

## 8. 1170을 위한 다음 우선순위

1. **독립 팀 OOF 확보와 constrained blend**: 현재 가장 큰 미개척 축이다. 같은 champion
   계보가 아닌 팀원 모델의 exact row-aligned OOF를 받아 시간·월·R_CORE/R_ANCHOR/F 최악
   gain 제약을 걸어 혼합한다.
2. **train-only 월별 base-rate forecast**: 월 숫자 자체를 test에서 보정하지 말고, 각
   origin 이전 시즌만으로 계절성·최근 추세를 예측하는 hierarchical logistic offset을
   사전 등록해 평가한다. 3월·10월 표본 부족은 partial pooling으로 처리한다.
3. **TrackMan 추가 탐색은 새 사전 가설만 허용**: IVB·고차원 물리·repeatability·구종
   학생 family는 닫는다. 새 물리 가설은 야구 메커니즘을 먼저 명시하고 오래된 두 독립
   origin을 통과할 때만 2024를 연다.
4. **위치 SOTA는 residual 구조만 이식**: 현재 pitch 물리량을 상상해 생성하지 않는다.
   pitcher×예측 구종×타자손의 shrinkage, count/context interaction을 train-only
   cross-fitting으로 구현하고 기존 모델과 잔차 상관이 낮을 때만 유지한다.

1170은 현재 점수에서 약 `+11.93`이 필요하다. 최근축의 local gain을 Public 점수 증가로
동일시할 수 없으며, 단일 미세 보정보다 독립 기반 모델의 error diversification이 목표에
도달할 가능성이 높다.

## 재현 위치

- 데이터 전수조사: `src/v65_trackman_data_census.py`
- 127개 profile 조사: `src/v67_trackman_command_proxy_census.py`
- IVB calendar gate: `src/v70_trackman_ivb_calendar_gate.py`
- 고정 recipe 과거축 반증: `src/v74_trackman_ivb_older_axes.py`
- pitch label 감사/재학습: `src/v72_trackman_pitch_type_label_audit.py`,
  `src/v73_trackman_label_pitch_type_student.py`
- 원시 요약: `reports/trackman_census_20260822_01/`,
  `reports/trackman_command_proxy_census_20260822_01/`,
  `reports/trackman_pitch_type_label_audit_20260822_01/`
