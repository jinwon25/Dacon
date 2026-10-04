# 상위권 입상을 위한 현황 및 후속 전략 (2026-08-08)

## 최신 공개 점수 업데이트 (2026-08-09)

`submission3 edit`(ID `39023`, `submit_v2.zip`, `schedule`, 2026-08-08 16:08:48)은 Public **763.2665303697**, 실행 시간 **9초**를 기록했다. 기존 incumbent 749.5249490965 대비 **+13.7415812732점(+1.83%)** 상승이다. 따라서 5% Trackman을 섞은 hybrid 방향은 배포 평가 구간에서도 유효하다는 강한 증거를 얻었다. 단, 단일 공개 점수로 두 구성요소의 기여를 분리할 수 없으므로 공개 점수에 맞춘 연속 튜닝은 하지 않고 사전 계산한 weight grid의 10% 지점만 다음 통제 후보로 승격했다.

| candidate | Trackman weight | local recency-weighted delta | final ZIP | SHA-256 |
|---|---:|---:|---|---|
| submitted `submit_v2.zip` | 5% | -0.000046588 | `submit_v2.zip` | `FE368AF109EF0BB8103D728F45794A6192599B691BF08DF02A22F31BD2D438A7` |
| next probe `submit_v3.zip` | 10% | -0.000071654 | `submit_v3.zip` | `6EE2CF45679F94C457A13B4FEFDCCADF3DF71E09B36F294CCF2A0EC6689C4579` |

`submit_v3.zip`은 v2의 모델 파일을 그대로 상속하고 `model/hybrid.json`의 `trackman_weight`만 0.10으로 변경했다. 파일명은 확장자를 포함해 13자로, 프로젝트의 40자 제한을 만족한다. 검증 결과 공식 sample 실행, 행 순서·확률 범위·오프라인 검사, 245,789행 대표 추론(13.621초, peak RSS 1,281.6MB)이 모두 통과했다.

## 결론

incumbent Public 749.524949는 보존하되, 실제 최신 기준점은 제출된 `submit_v2.zip`의 Public 763.2665303697이다. `hybrid_recency_r_trackman_w05_v1`은 사전 게이트를 통과했고, 그 공개 검증을 바탕으로 Trackman 비중 10%의 `submit_v3.zip`을 통제 후보로 별도 패키징했다.

| validation | incumbent Brier | candidate Brier | delta |
| ---: | ---: | ---: | ---: |
| 2021 | 0.246945046 | 0.246865966 | -0.000079080 |
| 2022 | 0.244169346 | 0.244151170 | -0.000018176 |
| 2023 | 0.251137482 | 0.251047904 | -0.000089578 |
| 2024 | 0.248460961 | 0.248440533 | -0.000020428 |

- 고정 recency weight 1:2:3:4 delta: **-0.000046588**
- worst-fold delta: **-0.000018176**
- 결합 pitcher-season cluster bootstrap P(improve): **1.0000**
- 2024 단독 cluster bootstrap P(improve): **0.8958**, 95% 구간은 0을 포함한다.
- 따라서 Public 제출 가치는 있지만, 독립 검증에서 확정된 큰 개선으로 과장하면 안 된다.

## 공식 대회 현황

2026-08-08 확인 기준 참가자 1,645명, 종료까지 25일이다. 팀 병합 마감은 8월 26일, 제출 마감은 9월 1일, 대회 종료는 9월 2일, 코드·PPT 마감은 9월 7일, Phase 3 발표는 9월 14일이다. 최신 일정은 [공식 개요](https://dacon.io/competitions/official/236743/overview/description)를 기준으로 다시 확인해야 한다.

Public 1위는 1,126.33191, 100위는 967.31541이다. incumbent와의 차이는 각각 376.81점, 217.79점이다. 순위와 점수는 계속 변하므로 [공식 리더보드](https://dacon.io/competitions/official/236743/leaderboard)를 단일 기준으로 사용한다.

평가지표는 Brier Skill Score이며 `max(0, 100000 * (1 - BS / (r * (1-r))))`이다. Public은 평가 데이터 전체를 사용하고 종료 시 Private로 전환된다. 상세 산식은 [공식 평가 페이지](https://dacon.io/competitions/official/236743/overview/evaluation)를 따른다.

중요 규칙은 다음과 같다.

- 외부 데이터와 외부 API는 금지다. 외부 논문은 방법론 선택에만 사용한다.
- 각 test 행은 독립적으로 추론해야 하며 다른 test 행의 분포·빈도·순서를 사용하지 않는다.
- Python 3.11.15, 6 CPU, 28GB RAM, L4 22.4GB, 설치 10분·추론 10분의 오프라인 환경이다.
- 하루 제출은 최대 5회이며, 약 100팀이 코드·PPT·재현 검증을 거쳐 오프라인 단계로 선발된다.

최신 제한은 반드시 [공식 규칙](https://dacon.io/competitions/official/236743/overview/rules)과 [공식 데이터 설명](https://dacon.io/competitions/official/236743/data)에서 재확인한다. 현재 [코드 공유](https://dacon.io/competitions/official/236743/codeshare)에는 공식 RF 학습·추론 베이스라인 외에 상위권 해법을 드러내는 커뮤니티 코드가 없다.

## 새로 확인한 데이터 구조

기존 감사의 `pitcher_id`와 `pitcher_trackman_id` 직접 교집합 0%는 사실이지만, 이것이 선수 연결 불가능을 뜻하지는 않았다. 메인 데이터의 누적 fastball/breaking/offspeed 비율과 누적 투구 수에서 연도별 구종 궤적을 복원한 뒤, 손잡이별 Hungarian 일대일 배정을 적용하면 반복적으로 같은 선수 쌍이 나온다.

- 2024→2025 연결 유지율: high 95.65%, medium 79.75%, low 69.94%, 전체 80.04%
- 2024 검증 행 프로필 커버리지: 전체 연결 약 64.4%, medium/high 약 49.6%, high 약 25.2%
- 고신뢰 연결에서 팀 소속의 Trackman 팀→익명 메인 팀 column-majority 일치율은 약 98.0%였다.
- 연결·프로필은 각 forecast origin 이전 시즌만 사용하며 target은 전혀 사용하지 않는다.

이 문제는 두 중복 없는 소스 사이의 clean-clean entity resolution이며, 전역 일대일 제약을 적용한 이유는 [one-to-one entity-resolution 알고리즘 비교 연구](https://biblio.ugent.be/publication/01KJT5QHPJ3PKSCJ5Q73Q1R470)의 문제 설정과 일치한다.

Trackman에는 plate location과 포수의 요구 위치가 없다. 따라서 release speed/spin/break/release point만으로 제구를 완전히 복원할 수 없다. 실제 연구에서도 릴리스 각도·속도·스핀축·수평 릴리스 위치가 투구 위치에 영향을 주지만 관계가 투수마다 다르게 나타난다([Influence of Release Parameters on Pitch Location](https://pmc.ncbi.nlm.nih.gov/articles/PMC7739723/)). 의도와 실행을 분리한 개인화 xCTRL이 기존 지표보다 안정적·예측력이 높았다는 연구도 있지만, 이 대회 Trackman에는 실제 위치와 의도 위치가 없으므로 xCTRL 자체를 구현할 수 없다([xCTRL 논문](https://arxiv.org/abs/2508.19184)). 이 때문에 Trackman은 주력 모델이 아니라 5% diversity 모델로 제한했다.

## 이번에 종료한 방향

1. 고정 rolling-damped drift 단독 후보
   - recency 평균은 좋아졌지만 2024 delta가 +0.000012로 최신연도 게이트에 실패했다.

2. Trackman LGB 단독 후보
   - raw/rolling 보정 후에도 2022–2024에서 incumbent보다 약했다.
   - 다만 incumbent와의 오차 다양성 때문에 1–10% 저가중치 블렌드는 네 연도 모두 개선했다.

3. 전체 시간감쇠 모델
   - 2024 RF는 반감기 1년에서 -0.000095 개선했지만 2023에서는 +0.000164 악화했다.
   - 2025 Public에서 F 전용 변화점 보정이 실패한 사실을 반영해, 감쇠 RF는 안정적인 `game_type=R`에만 적용했다.

4. 이미 충분히 기각된 계열
   - CatBoost, XGBoost, L2 GBDT, standalone hierarchical prior, RF 대형 하이퍼파라미터 탐색은 기존 2024 검증에서 약했다.
   - CatBoost의 ordered statistics가 범주형 target leakage를 줄인다는 원리는 타당하지만([NeurIPS 논문](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)), 이 데이터에서 실행한 세 설정은 모두 초기 반복에서 멈추고 incumbent를 크게 밑돌았다.

## 새 후보의 고정 레시피

1. incumbent의 engineered LGB 35% + official RF 65%와 기존 train-only logit offset을 유지한다.
2. `game_type=R` 행에서만 official RF를 시즌 반감기 1년 sample weight로 재학습한 RF로 교체한다.
3. 익명 투수 연결로 만든 prior-season Trackman 물리 프로필 LGB를 별도로 학습한다.
4. Trackman LGB에는 `damped_3_0.5`와 `damped_3_0.8` 50:50 rolling forecast를 적용한다.
5. 최종 확률은 R-감쇠 후보 95% + Trackman 후보 5%다.

5%는 결과 확인 전에 고정해 평가했으며 이후 Public 결과에 맞춰 조정하지 않는다. 시간감쇠는 느린 concept drift에 대응하는 일반적 방법이지만 하나의 고정 decay가 모든 관측치에 최적이지 않을 수 있다는 점은 [instance-conditional decay 연구](https://ojs.aaai.org/index.php/AAAI/article/view/29173)와 이번 연도별 결과가 모두 보여준다.

## 패키지 검증

- 제출된 v2 ZIP: `submit_v2.zip` (SHA-256 `FE368AF109EF0BB8103D728F45794A6192599B691BF08DF02A22F31BD2D438A7`)
- 다음 통제 후보: `submit_v3.zip` (SHA-256 `6EE2CF45679F94C457A13B4FEFDCCADF3DF71E09B36F294CCF2A0EC6689C4579`)
- incumbent SHA-256: `77F96448A7149F385F3A800E9238957FD9940C434AD73F0EE14EB0DA6D47AB60`
- 압축 크기: 8.469MB, 압축 해제 10.427MB
- v2 실제 압축 모델 245,789행 추론: 9.481초, peak RSS 1,234.6MB
- v3 실제 압축 모델 245,789행 추론: 13.621초, peak RSS 1,281.6MB
- 공식 5행 실행, row 순서, numeric/finite/[0,1], 오프라인, test batch independence 모두 통과

## 점수 격차 해석

`r(1-r)≈0.25`로 근사하면 100위와의 217.79점 차이는 Brier 약 0.000544, 1위와의 376.81점 차이는 약 0.000942다. 새 후보의 최신연도 개선 0.000020과 4-fold recency 개선 0.000047은 대략 한 자릿수\~20점 규모의 신호다. 분명한 전진이지만 이 후보 하나로 입상권 격차를 닫는다고 기대하면 안 된다.

## 이후 우선순위

1. **후보 1회 Public 제출**
   - incumbent는 계속 보존한다.
   - 새 후보는 복합 변경이므로 점수가 오르더라도 Trackman과 R-감쇠의 개별 Public 기여는 식별되지 않는다.
   - 점수가 내려가면 가중치 미세조정 없이 폐기한다.

2. **Trackman team-aware linkage v2**
   - 고신뢰 선수 쌍으로 연도별 팀 correspondence를 추정하고, 팀 이동 일치율을 배정 비용에 추가한다.
   - 구종군별 release-height/side와 movement의 within-pitch-type 표준편차를 추가한다. 전체 표준편차는 레퍼토리 혼합과 실제 일관성을 혼동한다.
   - 연결 안정성·2024 효과가 동시에 개선될 때만 새 모델을 학습한다.

3. **TabM 단일 고위험·고수익 실험**
   - 시간 분할에서는 방법 순위가 random split과 달라질 수 있고 단순 MLP/GBDT가 강하다는 [TabReD 연구](https://arxiv.org/abs/2406.19380)를 따라 2023·2024 time holdout만 우선 평가한다.
   - parameter-efficient ensemble인 [TabM](https://arxiv.org/abs/2410.24210)은 현재 tree ensemble과 다른 오차를 만들 가능성이 있다.
   - 로컬 GPU가 없어 CPU 학습비용이 크므로, 작은 feature/embedding 구성의 2024 파일럿이 incumbent와 convex-blend 가능성을 보일 때만 전체 폴드로 확장한다.

4. **제출 운영 원칙**
   - 하루 5회를 가중치 탐색에 쓰지 않는다.
   - 하나의 제출은 하나의 사전 문서화된 가설을 검증한다.
   - Public 점수와 ZIP hash를 `reports/submissions.csv`에 즉시 기록한다.
   - Public 개선 뒤에도 코드 재현·10분 추론·독립행 규칙을 통과하지 못하면 입상 후보로 취급하지 않는다.

## 해석상 주의

2021–2024 outer fold는 누적 148개 이상의 기존 실험과 이번 실험에 반복 사용되어 사실상 개발 데이터다. bootstrap은 관측 오차의 재표본 안정성을 보여줄 뿐 모델·가중치 선택 편향을 제거하지 않는다. 새 후보의 최종 판단에는 단 한 번의 Public deployment probe가 필요하다.
