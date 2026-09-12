# 후속 실험 문헌 검토

검토일: 2026-08-05
범위: 논문 원문, 학회 proceedings, 저자 원고, 공식 논문 페이지 30건. 논문에서 사용한 외부 데이터는 가져오지 않으며, 아래 결정은 공식 대회 데이터에 적용할 방법론만을 대상으로 한다.

## 적용 원칙 요약

- 시간 순방향 outer fold를 모델 선택의 기준으로 두고 random split 순위는 사용하지 않는다.
- Brier Score가 strictly proper score라는 점에 맞춰 calibration과 resolution을 함께 평가한다.
- calibration·blend weight는 평가할 outer year보다 앞선 순방향 OOF에서만 학습한다.
- 고카디널리티 entity는 CatBoost ordered statistics와 train-only empirical-Bayes backoff를 각각 검증한다.
- 6개 시즌의 base rate에는 복잡한 시계열 모델보다 last value, 짧은 평균, logit trend, damped trend와 단순 조합을 사용한다.
- xCTRL과 TrackMan 위치 모델은 필요한 의도 위치와 공식 선수 조인 키가 없어 직접 적용하지 않는다.

## 1. TabReD

- citation: [Rubachev et al., TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular DL Benchmarks](https://proceedings.iclr.cc/paper_files/paper/2025/file/571799482291411607c54984153190b0-Paper-Conference.pdf)
- venue 및 연도: ICLR 2025.
- 연구 문제: 산업형 정형 데이터 벤치마크가 시간 변화와 풍부한 피처를 충분히 반영하는가.
- 핵심 방법: 시간 분할을 갖는 대규모 산업형 데이터에서 GBDT와 여러 tabular DL 방법을 재평가.
- 논문이 실제로 검증한 주장: random split과 time split은 모델 순위를 바꿀 수 있고, 시간 변화·풍부한 피처 조건에서 GBDT와 단순 embedding MLP가 강한 기준선이었다.
- 현재 데이터와의 공통점: 대규모 정형 데이터이며 season drift와 engineered as-of 피처가 존재한다.
- 현재 데이터와의 차이: 여기서는 시즌이 6개뿐이고 야구 entity 구조가 강하다.
- 대회 규칙 적합성: 적합. 검증 설계 원칙만 적용한다.
- 적용할 실험 ID: WF-INFRA, W0-W5.
- 예상 효과: 선택 편향 감소와 2025 일반화 가능성 개선.
- 계산 비용: 4개 outer fold만큼 증가.
- 누수 위험: 낮음. split 구현 오류는 테스트로 차단한다.
- 결정: 채택. random split 결과를 제출 게이트에 사용하지 않는다.

## 2. 구조화된 데이터의 교차검증

- citation: [Roberts et al., Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure](https://www.wsl.ch/lud/biodiversity_events/papers/Roberts_et_al-2017-Ecography.pdf)
- venue 및 연도: Ecography 2017.
- 연구 문제: 관측치가 독립 동일분포가 아닐 때 일반 CV가 왜 낙관적인가.
- 핵심 방법: 시간·공간·계층 구조에 맞춘 blocking과 예측 목적에 맞는 holdout 설계.
- 논문이 실제로 검증한 주장: 의존 구조와 예측 대상이 맞지 않는 CV는 일반화 오차를 과소평가할 수 있다.
- 현재 데이터와의 공통점: pitcher, pitcher-season, season 계층과 시간 의존이 있다.
- 현재 데이터와의 차이: exact game ID가 없어 game blocking은 증명할 수 없다.
- 대회 규칙 적합성: 적합.
- 적용할 실험 ID: WF-INFRA, BOOTSTRAP.
- 예상 효과: fold 간 재현성과 uncertainty 해석 개선.
- 계산 비용: cluster bootstrap 중간.
- 누수 위험: 낮음.
- 결정: 채택. season-forward split과 pitcher/pitcher-season cluster bootstrap을 사용한다.

## 3. 강한 ERM 기준선

- citation: [Gulrajani and Lopez-Paz, In Search of Lost Domain Generalization](https://arxiv.org/abs/2007.01434)
- venue 및 연도: ICLR 2021.
- 연구 문제: domain generalization 기법이 공정한 모델 선택 아래 ERM을 안정적으로 이기는가.
- 핵심 방법: 동일한 모델 선택 절차로 여러 DG 알고리즘과 ERM 비교.
- 논문이 실제로 검증한 주장: 적절히 선택한 강한 ERM 기준선이 여러 복잡한 DG 방법과 경쟁적이었다.
- 현재 데이터와의 공통점: 시즌을 domain으로 볼 수 있다.
- 현재 데이터와의 차이: 이미지 벤치마크가 중심이며 현재 문제는 binary probability estimation이다.
- 대회 규칙 적합성: 원칙은 적합.
- 적용할 실험 ID: W0, W4.
- 예상 효과: 불필요한 domain 알고리즘 탐색 방지.
- 계산 비용: 절감.
- 누수 위험: 낮음.
- 결정: 채택. 먼저 incumbent ERM을 다년 재현한다.

## 4. GroupDRO

- citation: [Sagawa et al., Distributionally Robust Neural Networks for Group Shifts](https://arxiv.org/abs/1911.08731)
- venue 및 연도: ICLR 2020.
- 연구 문제: 알려진 group shift에서 worst-group 성능을 개선하는 조건.
- 핵심 방법: GroupDRO에 regularization과 조기 종료를 결합.
- 논문이 실제로 검증한 주장: 충분한 regularization 없이 GroupDRO의 worst-group 일반화가 나쁠 수 있으며 강한 regularization이 중요하다.
- 현재 데이터와의 공통점: season 및 entity sample-size group이 있다.
- 현재 데이터와의 차이: 평가 분포의 group mixture를 모르며 목표는 평균 Brier다.
- 대회 규칙 적합성: 구현 가능하지만 우선순위가 낮다.
- 적용할 실험 ID: W6-OPTIONAL.
- 예상 효과: 최악 시즌 안정성 가능성.
- 계산 비용: 높음.
- 누수 위험: group 정의를 결과에 맞추면 선택 편향.
- 결정: 보류. 단순 recency weighting과 ERM이 실패할 때만 검토한다.

## 5. CatBoost

- citation: [Prokhorenkova et al., CatBoost: Unbiased Boosting with Categorical Features](https://papers.nips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features.pdf)
- venue 및 연도: NeurIPS 2018.
- 연구 문제: boosting의 prediction shift와 target-statistics 누수를 줄이며 범주형을 처리하는 방법.
- 핵심 방법: ordered boosting과 ordered target statistics.
- 논문이 실제로 검증한 주장: 제안한 ordered 절차가 target leakage 편향을 완화하고 여러 정형 벤치마크에서 경쟁력 있었다.
- 현재 데이터와의 공통점: pitcher, batter, team 등 고카디널리티 범주형이 많다.
- 현재 데이터와의 차이: CSV 행 순서가 시간 순서라는 근거가 없다.
- 대회 규칙 적합성: native categorical은 적합하나 has_time은 부적합.
- 적용할 실험 ID: W3-CAT.
- 예상 효과: 임의 수치 ID 처리보다 entity 상호작용 개선.
- 계산 비용: 중간에서 높음.
- 누수 위험: validation target이 CTR 생성에 들어가면 높음.
- 결정: 채택. validation Pool을 분리하고 has_time=False로 검증한다.

## 6. LightGBM

- citation: [Ke et al., LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf)
- venue 및 연도: NeurIPS 2017.
- 연구 문제: 대규모 데이터에서 GBDT 학습을 효율화하는 방법.
- 핵심 방법: GOSS와 EFB.
- 논문이 실제로 검증한 주장: 제안한 표본·피처 축소가 여러 데이터에서 정확도 손실을 작게 유지하면서 학습을 가속했다.
- 현재 데이터와의 공통점: 147만 행 정형 데이터.
- 현재 데이터와의 차이: 현재 피처 수가 상대적으로 작아 EFB 이득은 제한적일 수 있다.
- 대회 규칙 적합성: 적합.
- 적용할 실험 ID: W0-LGB, W4-LGB.
- 예상 효과: 강한 속도 대비 성능 기준선.
- 계산 비용: 낮음에서 중간.
- 누수 위험: categorical/target encoding 구현에 좌우됨.
- 결정: 채택.

## 7. XGBoost

- citation: [Chen and Guestrin, XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- venue 및 연도: KDD 2016.
- 연구 문제: 정규화된 tree boosting을 대규모 환경에서 확장하는 방법.
- 핵심 방법: second-order objective, sparsity-aware split, 시스템 최적화.
- 논문이 실제로 검증한 주장: 여러 대규모 문제에서 확장성과 예측 성능을 보였다.
- 현재 데이터와의 공통점: 대규모 binary classification.
- 현재 데이터와의 차이: native high-cardinality categorical 실험의 중심은 아니다.
- 대회 규칙 적합성: 적합.
- 적용할 실험 ID: W4-XGB.
- 예상 효과: LightGBM과 다른 오류 구조 및 squared-error 비교.
- 계산 비용: 중간에서 높음.
- 누수 위험: 낮음.
- 결정: 조건부 채택. 설치 버전과 CPU 시간이 허용될 때 목적함수 비교에 포함한다.

## 8. FT-Transformer와 강한 단순 기준선

- citation: [Gorishniy et al., Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959)
- venue 및 연도: NeurIPS 2021.
- 연구 문제: tabular DL 아키텍처를 공정하게 비교하고 단순한 강한 모델을 설계하는 방법.
- 핵심 방법: ResNet-like MLP와 FT-Transformer를 여러 데이터에서 GBDT와 비교.
- 논문이 실제로 검증한 주장: 단일 모델이 모든 정형 데이터에서 지배적이지 않고, 강한 단순 DL 기준선이 필요하다.
- 현재 데이터와의 공통점: 수치·범주 혼합 정형 데이터.
- 현재 데이터와의 차이: 제출 CPU 추론과 패키지 설치 제약이 강하다.
- 대회 규칙 적합성: 적합하지만 후순위.
- 적용할 실험 ID: W6-OPTIONAL.
- 예상 효과: GBDT와 다른 오류 다양성.
- 계산 비용: 높음.
- 누수 위험: 낮음.
- 결정: 보류.

## 9. tree inductive bias

- citation: [Grinsztajn et al., Why do tree-based models still outperform deep learning on typical tabular data?](https://arxiv.org/abs/2207.08815)
- venue 및 연도: NeurIPS 2022.
- 연구 문제: 전형적 중간 규모 정형 데이터에서 tree가 DL보다 강한 이유.
- 핵심 방법: 데이터 특성을 통제한 대규모 benchmark와 inductive-bias 분석.
- 논문이 실제로 검증한 주장: 비매끄러운 함수, 비정보 피처, 회전 비불변성 등에서 tree의 장점이 나타났다.
- 현재 데이터와의 공통점: heterogeneous feature와 임계값 상호작용이 많다.
- 현재 데이터와의 차이: 표본 수가 더 크고 고카디널리티 entity가 강하다.
- 대회 규칙 적합성: 적합.
- 적용할 실험 ID: W0-W4.
- 예상 효과: GBDT 우선 자원 배분의 근거.
- 계산 비용: 절감.
- 누수 위험: 낮음.
- 결정: 채택.

## 10. numerical feature embeddings

- citation: [Gorishniy et al., On Embeddings for Numerical Features in Tabular Deep Learning](https://papers.neurips.cc/paper_files/paper/2022/file/9e9f0ffc3d836836ca96cbf8fe14b105-Paper-Conference.pdf)
- venue 및 연도: NeurIPS 2022.
- 연구 문제: 수치형 피처의 표현을 학습해 tabular DL을 개선할 수 있는가.
- 핵심 방법: piecewise-linear encoding과 periodic activation.
- 논문이 실제로 검증한 주장: 여러 benchmark에서 적절한 수치 embedding이 MLP/Transformer를 개선했다.
- 현재 데이터와의 공통점: count, inning, as-of rate 등 수치 피처가 있다.
- 현재 데이터와의 차이: GBDT의 계산 효율이 훨씬 중요하다.
- 대회 규칙 적합성: 적합.
- 적용할 실험 ID: W6-MLP-PLR.
- 예상 효과: ensemble diversity.
- 계산 비용: 높음.
- 누수 위험: 낮음.
- 결정: 보류.

## 11. TabM

- citation: [Gorishniy et al., TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://arxiv.org/abs/2410.24210)
- venue 및 연도: ICLR 2025.
- 연구 문제: MLP ensemble의 이득을 파라미터 효율적으로 얻는 방법.
- 핵심 방법: 대부분의 파라미터를 공유하는 다중 MLP 예측.
- 논문이 실제로 검증한 주장: 공개 benchmark에서 개별 약한 구성원의 결합이 강한 tabular DL 기준선을 만들었다.
- 현재 데이터와의 공통점: ensemble diversity가 중요하다.
- 현재 데이터와의 차이: 147만 행 전체 학습·패키지 시간 검증이 필요하다.
- 대회 규칙 적합성: 적합.
- 적용할 실험 ID: W6-TABM.
- 예상 효과: GBDT 상관도가 낮은 보조 예측.
- 계산 비용: 높음.
- 누수 위험: 낮음.
- 결정: 보류. Wave 0–5 이후 자원이 남을 때만 실행한다.

## 12. Better by Default

- citation: [Holzmüller et al., Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data](https://arxiv.org/abs/2407.04491)
- venue 및 연도: NeurIPS 2024.
- 연구 문제: 광범위한 튜닝 없이도 강한 MLP와 boosted-tree default를 만들 수 있는가.
- 핵심 방법: meta-train 데이터에서 default를 선택하고 분리된 meta-test에서 평가.
- 논문이 실제로 검증한 주장: 개선된 default의 GBDT와 RealMLP 결합이 유리한 성능-시간 절충을 보였다.
- 현재 데이터와의 공통점: 목적 있는 소수 설정만 비교한다.
- 현재 데이터와의 차이: 현재 데이터의 temporal split에 맞춘 default 재검증이 필요하다.
- 대회 규칙 적합성: 적합.
- 적용할 실험 ID: W3, W4, W6.
- 예상 효과: hyperparameter overfit 감소.
- 계산 비용: 절감.
- 누수 위험: 낮음.
- 결정: 채택. 4–8개 이내 사전 정의 설정만 사용한다.

## 13. supervised probability calibration

- citation: [Niculescu-Mizil and Caruana, Predicting Good Probabilities with Supervised Learning](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf)
- venue 및 연도: ICML 2005.
- 연구 문제: 서로 다른 classifier의 확률 왜곡과 post-hoc calibration 효과.
- 핵심 방법: Platt scaling과 isotonic regression을 독립 calibration 자료에서 비교.
- 논문이 실제로 검증한 주장: 모델 종류별 확률 왜곡이 다르고, isotonic은 유연하지만 소표본에서 더 쉽게 과적합했다.
- 현재 데이터와의 공통점: RF와 boosted tree 확률을 결합한다.
- 현재 데이터와의 차이: temporal drift가 있어 random CV calibration은 부적합하다.
- 대회 규칙 적합성: nested OOF에서만 적합.
- 적용할 실험 ID: W1-CAL.
- 예상 효과: reliability 개선 가능성.
- 계산 비용: 낮음.
- 누수 위험: 같은 validation에서 fit/evaluate하면 높음.
- 결정: 채택. identity를 포함하고 순방향 OOF로만 fit한다.

## 14. Beta calibration

- citation: [Kull et al., Beta calibration: a well-founded and easily implemented improvement on logistic calibration](https://proceedings.mlr.press/v54/kull17a.html)
- venue 및 연도: AISTATS 2017.
- 연구 문제: logistic calibration이 identity map을 표현하지 못하는 문제.
- 핵심 방법: beta distribution에서 유도한 log(p), log(1-p) 기반 calibration family.
- 논문이 실제로 검증한 주장: identity를 포함하는 더 유연한 family가 여러 binary classifier에서 logistic calibration의 실패를 보완했다.
- 현재 데이터와의 공통점: incumbent의 Platt가 악화됐고 identity-safe 후보가 필요하다.
- 현재 데이터와의 차이: 시즌 drift 때문에 계수의 시간 안정성이 추가로 필요하다.
- 대회 규칙 적합성: 적합.
- 적용할 실험 ID: W1-BETA.
- 예상 효과: 과도한 slope 변경 없이 비대칭 왜곡 보정.
- 계산 비용: 매우 낮음.
- 누수 위험: calibration fold 분리 실패 시 높음.
- 결정: 채택. L2 규제와 identity 후보를 함께 비교한다.

## 15. strictly proper scoring rules

- citation: [Gneiting and Raftery, Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf)
- venue 및 연도: JASA 2007.
- 연구 문제: 확률 예측을 정직하게 평가하는 scoring rule의 이론.
- 핵심 방법: proper scoring rule의 표현과 entropy/divergence 연결.
- 논문이 실제로 검증한 주장: proper score는 예측자가 진실한 분포를 보고하도록 유도하며 Brier는 대표적 proper score다.
- 현재 데이터와의 공통점: 최종 지표가 Brier 기반이다.
- 현재 데이터와의 차이: 실제 ranking은 Brier Skill로 affine 변환된다.
- 대회 규칙 적합성: 적합.
- 적용할 실험 ID: METRIC, W1-W5.
- 예상 효과: accuracy 대신 probability quality에 모델 선택 정렬.
- 계산 비용: 없음.
- 누수 위험: 없음.
- 결정: 채택.

## 16. stable reliability diagrams

- citation: [Dimitriadis et al., Stable reliability diagrams for probabilistic classifiers](https://pmc.ncbi.nlm.nih.gov/articles/PMC7923594/)
- venue 및 연도: PNAS 2021.
- 연구 문제: 임의 bin 선택에 민감한 reliability diagram을 안정화하는 방법.
- 핵심 방법: PAV 기반 CORP diagram과 score decomposition.
- 논문이 실제로 검증한 주장: 재현 가능하고 통계적으로 일관된 reliability 진단과 uncertainty quantification을 제공한다.
- 현재 데이터와의 공통점: calibration curve와 Brier decomposition이 필요하다.
- 현재 데이터와의 차이: 제출 선택은 Brier 자체이며 CORP는 주로 진단에 쓴다.
- 대회 규칙 적합성: validation 진단에 적합.
- 적용할 실험 ID: DIAGNOSTICS.
- 예상 효과: 임의 10-bin 결론의 민감도 감소.
- 계산 비용: 낮음.
- 누수 위험: validation target은 진단에만 사용해야 한다.
- 결정: 채택. 고정 10/20-bin과 함께 진단하되 그 bin을 test 보정에 쓰지 않는다.

## 17. Super Learner

- citation: [van der Laan et al., Super Learner](https://biostats.bepress.com/ucbbiostat/paper222/)
- venue 및 연도: Statistical Applications in Genetics and Molecular Biology 2007.
- 연구 문제: 여러 학습기의 cross-validated risk를 이용한 결합.
- 핵심 방법: cross-validation 예측에서 convex combination을 선택하고 전체 데이터로 base learners 재학습.
- 논문이 실제로 검증한 주장: 후보 라이브러리와 조건 아래 oracle 성질을 갖는 cross-validated ensemble을 제시했다.
- 현재 데이터와의 공통점: RF, LGB, CatBoost, prior를 Brier로 결합한다.
- 현재 데이터와의 차이: IID CV 대신 season-forward OOF가 필요하다.
- 대회 규칙 적합성: nested temporal OOF로 변형하면 적합.
- 적용할 실험 ID: W5-SL.
- 예상 효과: 모델 다양성 활용과 수동 weight 편향 감소.
- 계산 비용: base OOF 생성이 큼, 최적화 자체는 낮음.
- 누수 위험: 같은 outer fold에서 weight 선택·평가하면 매우 높음.
- 결정: 채택. nonnegative sum-one와 L2 shrinkage를 nested로 평가한다.

## 18. exponential smoothing state space

- citation: [Hyndman et al., A State Space Framework for Automatic Forecasting Using Exponential Smoothing Methods](https://robjhyndman.com/papers/hksg.pdf)
- venue 및 연도: International Journal of Forecasting 2002.
- 연구 문제: exponential smoothing을 state-space model로 통합하고 자동 선택하는 방법.
- 핵심 방법: error, trend, seasonality 조합의 likelihood 기반 state-space 표현.
- 논문이 실제로 검증한 주장: 다양한 exponential smoothing 기법을 일관된 확률 모델과 forecast interval로 표현했다.
- 현재 데이터와의 공통점: season-level base-rate drift가 있다.
- 현재 데이터와의 차이: 관측점이 6개뿐이고 seasonality를 추정할 수 없다.
- 대회 규칙 적합성: 이전 시즌 rate만 사용하면 적합.
- 적용할 실험 ID: W1-DRIFT.
- 예상 효과: 선형 trend보다 안정적인 global offset 가능성.
- 계산 비용: 매우 낮음.
- 누수 위험: 2025 예측 평균을 fit에 쓰면 높음.
- 결정: 제한 채택. level/damped trend만 rolling-origin으로 비교한다.

## 19. damped trend

- citation: [Gardner and McKenzie, Damped Trend Exponential Smoothing](https://www.bauer.uh.edu/gardner/Damped-trend-Modelling.pdf)
- venue 및 연도: Management Science 계열 저자 원고; damped-trend 연구.
- 연구 문제: 장기 선형 trend의 과도한 외삽을 완화하는 방법.
- 핵심 방법: trend 성분을 미래 horizon에서 감쇠.
- 논문이 실제로 검증한 주장: 여러 시계열에서 damped trend가 무감쇠 trend보다 robust한 forecast를 제공할 수 있음을 보였다.
- 현재 데이터와의 공통점: 성공률 하락 추세를 2025로 한 단계 외삽한다.
- 현재 데이터와의 차이: 시즌 관측이 극소수다.
- 대회 규칙 적합성: 적합.
- 적용할 실험 ID: W1-DAMPED.
- 예상 효과: incumbent linear-logit trend의 과도한 이동 방지.
- 계산 비용: 매우 낮음.
- 누수 위험: damping을 2024 결과에 맞춰 사후 선택하면 중간.
- 결정: 채택. damping grid를 사전 고정하고 rolling-origin으로만 선택한다.

## 20. xCTRL

- citation: [Ludwig, Brill, and Wyner, Separating Intent from Execution: A Probabilistic Approach to Pitch Location Accuracy](https://arxiv.org/abs/2508.19184)
- venue 및 연도: 2025 preprint.
- 연구 문제: 투수별 의도 위치와 실제 실행을 분리해 control을 측정하는 방법.
- 핵심 방법: 개인화된 투수·상황별 의도 위치 분포와 실제 위치의 차이를 모델링.
- 논문이 실제로 검증한 주장: 저자 데이터에서 개인화된 xCTRL이 기존 control 지표보다 안정성과 예측력을 보였다.
- 현재 데이터와의 공통점: pitcher×pitch type×batter hand 개인화와 최소 표본 문제가 중요하다.
- 현재 데이터와의 차이: 현재 메인 데이터에는 의도 위치와 실제 위치가 없고 TrackMan ID 직접 조인도 0%다.
- 대회 규칙 적합성: 원칙만 적합; 수치·외부 데이터 결합은 금지.
- 적용할 실험 ID: W2-HIER.
- 예상 효과: 계층적 entity residual 설계의 근거.
- 계산 비용: 낮음에서 중간.
- 누수 위험: 현재 pitch 결과 위치를 쓰면 치명적.
- 결정: 부분 채택. 개별화·표본 신뢰도·backoff 원칙만 사용하고 xCTRL 자체는 기각한다.

## 21. pitch type/location prediction

- citation: [Lee, Prediction of pitch type and location in baseball using ensemble model of deep neural networks](https://journals.sagepub.com/doi/pdf/10.3233/JSA-200559)
- venue 및 연도: Journal of Sports Analytics 2022.
- 연구 문제: 경기 상황과 과거 투구로 다음 pitch type과 location을 공동 예측.
- 핵심 방법: deep neural network ensemble.
- 논문이 실제로 검증한 주장: 저자 데이터에서 ensemble이 다음 투구 유형·위치 예측에 사용 가능함을 보였다.
- 현재 데이터와의 공통점: count, handedness, pitcher/batter context가 있다.
- 현재 데이터와의 차이: 각 test 행은 독립이며 test 순서를 history로 사용할 수 없다.
- 대회 규칙 적합성: row-local/as-of context 원칙만 적합.
- 적용할 실험 ID: W2-CONTEXT.
- 예상 효과: count×hand×runner interaction 후보.
- 계산 비용: 낮음.
- 누수 위험: test 순서를 연속 투구로 사용하면 치명적.
- 결정: 부분 채택. 제공된 as-of row-local 피처와 train-only interaction만 사용한다.

## 22. TrackMan VBGMM

- citation: [Hirotsu et al., Application of VBGMM for pitch type classification: analysis of TrackMan pitch tracking data](https://link.springer.com/article/10.1007/s42081-020-00079-8)
- venue 및 연도: Japanese Journal of Statistics and Data Science 2020.
- 연구 문제: TrackMan 물리 피처로 pitch type을 확률적으로 군집화하는 방법.
- 핵심 방법: variational Bayesian Gaussian mixture.
- 논문이 실제로 검증한 주장: 저자 TrackMan 데이터에서 투구 유형 군집화와 불확실성 표현이 가능했다.
- 현재 데이터와의 공통점: 별도 TrackMan 로그가 있다.
- 현재 데이터와의 차이: 메인 선수 ID와 직접 조인 커버리지가 0%다.
- 대회 규칙 적합성: 공식 키가 확인될 때만 적합.
- 적용할 실험 ID: TRACKMAN-HOLD.
- 예상 효과: 현재는 없음.
- 계산 비용: 높음.
- 누수 위험: 퍼지 ID 매핑과 현재 투구 결과 사용 위험이 높음.
- 결정: 기각/보류. 공식 조인 증거 없이는 실행하지 않는다.

## 23. Brier decomposition

- citation: [Murphy, A New Vector Partition of the Probability Score](https://journals.ametsoc.org/view/journals/apme/12/4/1520-0450_1973_012_0595_anvpot_2_0_co_2.xml)
- venue 및 연도: Journal of Applied Meteorology 1973.
- 연구 문제: Brier score를 해석 가능한 구성요소로 분해하는 방법.
- 핵심 방법: uncertainty, reliability, resolution 분해.
- 논문이 실제로 검증한 주장: probability score의 세 구성요소와 해석을 정식화했다.
- 현재 데이터와의 공통점: 모델 개선이 calibration인지 resolution인지 구분해야 한다.
- 현재 데이터와의 차이: finite-bin 추정에는 표본·bin 민감도가 있다.
- 대회 규칙 적합성: validation 진단에 적합.
- 적용할 실험 ID: DIAGNOSTICS.
- 예상 효과: calibration-only 개선의 한계 파악.
- 계산 비용: 낮음.
- 누수 위험: 없음.
- 결정: 채택.

## 24. empirical-Bayes categorical encoding

- citation: [Micci-Barreca, A Preprocessing Scheme for High-Cardinality Categorical Attributes](https://dl.acm.org/doi/10.1145/507533.507538)
- venue 및 연도: SIGKDD Explorations 2001.
- 연구 문제: 고카디널리티 범주형을 안정적인 수치 표현으로 바꾸는 방법.
- 핵심 방법: empirical-Bayes shrinkage와 hierarchy blending.
- 논문이 실제로 검증한 주장: 범주별 관측 통계를 global/hierarchical 통계로 shrink하는 실용적 전처리 방식을 제시했다.
- 현재 데이터와의 공통점: pitcher/batter 및 sparse interaction.
- 현재 데이터와의 차이: season-forward OOF가 필수다.
- 대회 규칙 적합성: train-only/OOF로 만들면 적합.
- 적용할 실험 ID: W2-HIER.
- 예상 효과: cold-start와 소표본 variance 감소.
- 계산 비용: 중간.
- 누수 위험: training row에 자기 target이 포함되면 높음.
- 결정: 채택. outer train 내부도 season-forward OOF encoding을 사용한다.

## 25. isotonic probability calibration

- citation: [Zadrozny and Elkan, Transforming Classifier Scores into Accurate Multiclass Probability Estimates](https://www.cs.columbia.edu/~djhsu/ML/handouts/zadrozny2002kdd.pdf)
- venue 및 연도: KDD 2002.
- 연구 문제: classifier ranking score를 calibrated probability로 변환하는 방법.
- 핵심 방법: binary decomposition과 isotonic regression.
- 논문이 실제로 검증한 주장: 여러 분류 문제에서 isotonic 기반 score-to-probability 변환을 평가했다.
- 현재 데이터와의 공통점: binary probability 품질이 직접 점수다.
- 현재 데이터와의 차이: 시간 drift 아래 calibration map 안정성이 중요하다.
- 대회 규칙 적합성: 충분한 historical OOF에서만 적합.
- 적용할 실험 ID: W1-ISO.
- 예상 효과: 비선형 calibration 왜곡 보정.
- 계산 비용: 낮음.
- 누수 위험: outer validation으로 fit하면 치명적.
- 결정: 보조 채택. early fold 자료가 부족하면 진단만 한다.

## 26. forecast combination

- citation: [Bates and Granger, The Combination of Forecasts](https://www.tandfonline.com/doi/abs/10.1057/jors.1969.103)
- venue 및 연도: Operational Research Quarterly 1969.
- 연구 문제: 서로 다른 forecast를 결합해 MSE를 줄일 수 있는가.
- 핵심 방법: 과거 오차를 이용한 linear combination.
- 논문이 실제로 검증한 주장: airline-passenger 예시에서 composite forecast가 개별 forecast보다 낮은 MSE를 낼 수 있었다.
- 현재 데이터와의 공통점: squared loss인 Brier와 base-rate forecast 조합.
- 현재 데이터와의 차이: 시즌 rate 관측이 매우 적어 최적 weight 분산이 크다.
- 대회 규칙 적합성: rolling-origin weight만 적합.
- 적용할 실험 ID: W1-DRIFT-COMB, W5.
- 예상 효과: 특정 trend 가정 실패 완화.
- 계산 비용: 매우 낮음.
- 누수 위험: 사후 weight 조정 시 높음.
- 결정: 채택. 단순 평균과 강한 shrinkage를 우선한다.

## 27. temperature/logit scaling

- citation: [Guo et al., On Calibration of Modern Neural Networks](https://proceedings.mlr.press/v70/guo17a.html)
- venue 및 연도: ICML 2017.
- 연구 문제: 현대 신경망의 confidence calibration과 간단한 post-hoc 보정.
- 핵심 방법: temperature scaling을 포함한 calibration 비교.
- 논문이 실제로 검증한 주장: 저자 이미지·문서 데이터에서 단일 temperature가 여러 현대 신경망의 calibration을 효과적으로 개선했다.
- 현재 데이터와의 공통점: slope-only 또는 intercept-only처럼 강하게 규제된 보정이 필요하다.
- 현재 데이터와의 차이: GBDT/RF이며 주된 shift는 season base rate다.
- 대회 규칙 적합성: nested OOF에서 적합.
- 적용할 실험 ID: W1-PLATT.
- 예상 효과: 낮은 분산의 calibration baseline.
- 계산 비용: 매우 낮음.
- 누수 위험: calibration data 재사용 시 높음.
- 결정: 부분 채택. full Platt와 intercept-only를 분리 비교한다.

## 28. verified calibration

- citation: [Kumar et al., Verified Uncertainty Calibration](https://papers.neurips.cc/paper_files/paper/2019/hash/f8c0c968632845cd133308b1a494967f-Abstract.html)
- venue 및 연도: NeurIPS 2019.
- 연구 문제: calibration error 추정과 post-hoc calibrator의 표본 효율.
- 핵심 방법: scaling-binning calibrator와 calibration-error estimator 분석.
- 논문이 실제로 검증한 주장: 일반 scaling의 calibration 평가가 낙관적일 수 있고 histogram 계열은 더 많은 표본을 요구한다.
- 현재 데이터와의 공통점: calibration 성능을 같은 OOF에서 과대평가할 위험이 있다.
- 현재 데이터와의 차이: binary 대규모 데이터라 표본은 많지만 독립 시즌 수는 적다.
- 대회 규칙 적합성: 진단 원칙은 적합.
- 적용할 실험 ID: W1-CAL, BOOTSTRAP.
- 예상 효과: calibration 선택 편향 통제.
- 계산 비용: 낮음.
- 누수 위험: 높음 if calibration fit/eval reuse.
- 결정: 채택. outer-year nested 평가와 cluster bootstrap을 사용한다.

## 29. prior-probability shift correction

- citation: [Saerens, Latinne, and Decaestecker, Adjusting the Outputs of a Classifier to New a Priori Probabilities](https://dipot.ulb.ac.be/dspace/bitstream/2013/68391/1/Decaestecker_NeuralComp02.pdf)
- venue 및 연도: Neural Computation 2002.
- 연구 문제: class prior가 바뀔 때 classifier posterior를 재학습 없이 조정하는 방법.
- 핵심 방법: 새 prior에 맞춘 posterior correction과 EM 추정.
- 논문이 실제로 검증한 주장: prior-probability shift 가정 아래 posterior adjustment 절차를 유도하고 실험했다.
- 현재 데이터와의 공통점: 시즌별 global success rate drift가 크다.
- 현재 데이터와의 차이: test 분포로 새 prior를 추정하는 EM은 대회 규칙상 금지되고 label shift 가정도 검증되지 않았다.
- 대회 규칙 적합성: test-EM은 부적합; train-only forecast offset만 적합.
- 적용할 실험 ID: W1-OFFSET.
- 예상 효과: entity 순위를 유지하며 global mean drift 보정.
- 계산 비용: 매우 낮음.
- 누수 위험: test 예측 분포를 사용하면 치명적.
- 결정: 제한 채택. 이전 시즌 rate로 예측한 고정 logit offset만 사용하고 test-EM은 기각한다.

## 30. calibration hierarchy

- citation: [Van Calster et al., A calibration hierarchy for risk models was defined](https://pubmed.ncbi.nlm.nih.gov/26772608/)
- venue 및 연도: Journal of Clinical Epidemiology 2016.
- 연구 문제: calibration의 서로 다른 강도를 어떻게 구분해 외부 검증할 것인가.
- 핵심 방법: mean, weak, moderate, strong calibration의 계층적 정의와 simulation.
- 논문이 실제로 검증한 주장: 평균 예측 일치만으로 전체 calibration을 주장할 수 없고 intercept와 slope 등 여러 수준을 구분해야 한다.
- 현재 데이터와의 공통점: global base-rate offset과 slope distortion을 별도 진단해야 한다.
- 현재 데이터와의 차이: 임상 risk model이 아니라 야구 pitch probability다.
- 대회 규칙 적합성: 적합.
- 적용할 실험 ID: DIAGNOSTICS, W1.
- 예상 효과: base-rate correction을 전체 calibration 개선으로 과장하지 않음.
- 계산 비용: 낮음.
- 누수 위험: 없음.
- 결정: 채택. fold별 prediction mean gap, calibration intercept, slope를 모두 기록한다.

## 실행 우선순위

1. WF-INFRA와 W0로 incumbent의 다년 안정성을 먼저 확정한다.
2. W1에서 train-only base-rate forecast, intercept-only, Beta calibration을 nested 평가한다.
3. W2에서 empirical-Bayes hierarchy와 cold-start 진단을 수행한다.
4. W3 CatBoost, W4 logloss 대 L2를 소수 설정으로 비교한다.
5. W5는 모든 base prediction이 준비된 뒤에만 수행한다.
6. W6 TabM/MLP-PLR와 TrackMan은 앞선 단계의 개선 신호와 자원 여유가 있을 때만 재검토한다.
