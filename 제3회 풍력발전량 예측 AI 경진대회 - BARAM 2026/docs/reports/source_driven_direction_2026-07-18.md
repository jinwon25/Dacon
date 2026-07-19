# 공개 사례 기반 병목 진단 및 발전 방향 (2026-07-18)

## 결론

현재 공개 최고 제출 `blend_best_crossg3_traj_meta_finesweep.csv`의 총점은
`0.6417471627`이고, 구성 점수는 `1-nMAE=0.8754733572`,
`FiCR=0.4080209682`이다. 같은 1-nMAE를 유지한 채 총점 0.65를 달성하려면
FiCR가 약 `0.4245266428`까지 올라야 한다. 따라서 다음 병목은 평균 오차를
조금 더 줄이는 affine 보정보다, 6%/8% 오차 구간 안으로 들어오는 사례를
안정적으로 늘리는 것이다.

2026-07-18 기준 공식 리더보드의 상위권과 0.65 부근 제출도 대체로
FiCR `0.426~0.455`를 보인다. 우리 모델의 1-nMAE는 이미 경쟁 가능한 반면
FiCR가 상대적으로 낮다는 판단과 일치한다.

- 공식 리더보드: https://dacon.io/competitions/official/236727/leaderboard
- 공식 평가 산식: https://dacon.io/en/competitions/official/236727/codeshare/14035

## 공개 입상 사례에서 가져올 구조

### 1. KDD Cup 2022 1위: 상태·예보 구간별 동적 앙상블

1위 FDSTT는 하나의 학습형 stacking 모델에 모든 상황을 맡기지 않았다.
시공간 딥러닝 모델과 시간·공간을 분리한 LightGBM을 함께 사용하고,
현재 출력 상태와 예보 구간에 따라 구성 모델을 동적으로 선택했다.
분포가 불안정할 때 일반적인 stacking이 과적합할 수 있다는 문제의식도
명시했다.

- 대회 순위: https://baidukddcup2022.github.io/
- 1위 논문: https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_0518.pdf

현재 프로젝트에 대응시키면 행 단위 meta/affine가 아니라 `NWP issue × 6시간
lead block` 단위의 전문가 라우팅이다. 라우터 입력은 발전량 정답이나 SCADA가
아니라 제출 시점에 사용 가능한 NWP 궤적, 풍속·풍향, 기압, 돌풍, 모델 간
불일치만 사용해야 한다.

### 2. KDD Cup 2022 2위·7위·10위: 이질적 전문가와 시간 구간 분리

2위는 LightGBM, GRU, 로컬 앙상블을 서로 다른 입력 조합과 다단계 예측
방식으로 구성했다. 7위와 10위도 GBDT와 순환 신경망을 결합하고, 터빈·예보
구간·시간 척도 차이를 분리했다. 핵심은 같은 모델의 seed만 늘리는 것이 아니라
오류 구조가 다른 전문가를 만드는 것이다.

- 2위 논문: https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_1286.pdf
- 7위 논문: https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_1833.pdf
- 7위 코드: https://github.com/linfangquan/kddcup2022
- 10위 논문: https://baidukddcup2022.github.io/papers/Baidu_KDD_Cup_2022_Workshop_paper_0931.pdf
- 10위 코드: https://github.com/injadlu/KDDCUP2022

현재 보유한 spatiotemporal seed 전문가들은 어느 정도 다양성이 있지만 같은
NWP 계열과 학습 표면을 공유한다. 다음 독립 전문가는 KMA UM 같은 별도 운영
예보 또는 풍향 sector별 물리 피처를 사용해야 한다.

### 3. GEFCom: 발전소·예보시간별 모델과 유사 기상 라우팅

GEFCom 풍력 사례는 발전소와 예보시간별 모델, 기상 유사도 군집, 실제 제출
블록과 닮은 검증 설계를 사용했다. 우승 접근도 풍속 신호 평활화와 발전소 간
2단계 결합을 활용했다.

- GEFCom 2012 사례: https://www.sciencedirect.com/science/article/pii/S0169207013000836
- GEFCom 2014 우승 접근: https://www.sciencedirect.com/science/article/pii/S0169207016000145

## 로컬 검증 결과

### 정답을 아는 상한 진단

기존 H2 OOF 전문가 중 최적 전문가를 정답을 보고 고르는 진단에서 그룹 3의
점수 상한은 다음과 같았다. 이는 제출 가능한 모델이 아니라 탐색 가치만 확인하는
oracle이다.

| 라우팅 단위 | 그룹 3 점수 상한 증가 |
|---|---:|
| 24시간 issue | +0.06562 |
| 6시간 lead block | +0.09907 |

0.65에 필요한 그룹 3 개선량보다 상한이 크므로 동적 라우팅 자체는 유효한
탐색 공간이다.

- 진단 코드: `experiments/dynamic_router_oracle.py`
- 결과: `artifacts_final/diagnostics/dynamic_router_oracle_20260718.json`

### 정답을 보지 않는 기상 유사도 라우터

Q1의 과거 유틸리티만 사용해 Q2 정책을 고르고, 선택 정책을 고정한 뒤 H1으로
H2를 한 번만 평가했다. 6시간 블록의 9.9%를 라우팅했다.

| H2 지표 | 변화량 |
|---|---:|
| 총점 | +0.001709 |
| 1-nMAE | +0.000833 |
| FiCR | +0.002585 |
| 최악 월(2024-12) 총점 | -0.003540 |

전체 평균은 개선됐지만 12월 손실로 promotion gate를 통과하지 못했다.
제출 파일은 만들지 않았다. 이번 결과는 동적 라우팅의 일부 상한을 정답 없이
회수할 수 있음을 보여주는 동시에, 계절 전이와 최악 월 방어가 다음 병목임을
확인한다.

- 진단 코드: `experiments/weather_similarity_router.py`
- 결과: `artifacts_final/diagnostics/weather_similarity_router_20260718.json`

## 실행 우선순위

1. **강건한 블록 라우터**: 평균 유틸리티 대신 source-month/풍속구간별 하위
   분위 유틸리티를 사용한다. 한 전문가가 모든 관측 subgroup에서 양수일 때만
   라우팅하고, 나머지는 incumbent로 되돌린다.
2. **독립 NWP 전문가**: 규정상 허용되고 재현 가능한 KMA UM 운영 예보를 확보한
   뒤, 기존 LDAPS/GFS와 독립적인 전문가 및 disagreement 피처를 만든다.
3. **FiCR 경계 전용 학습**: 실제 발전량 기준 6%/8% 경계 부근 표본의 가중치를
   올리되, 월별 최악 성능과 1-nMAE 비열화를 동시에 제약한다.
4. **교차 발전소 결합은 보조로 제한**: GEFCom의 cross-sectional 아이디어는
   기존 cross-group 계열의 작은 blending/gating에만 사용한다. 대형 graph 모델
   교체는 이미 전이 실패가 있어 보류한다.

## 중단할 탐색

- 공개 점수를 따라가는 추가 affine/계절 배율 미세조정
- 단일 split에서만 좋아지는 row-wise meta gate
- 기존 NWP 입력을 그대로 쓰는 generic Transformer/graph 재학습
- 평균 CV만 보고 만드는 새 제출

새 후보는 총점·1-nMAE·FiCR가 모두 양수이고, 월/풍속/lead block 중 어느 핵심
subgroup에서도 유의미한 음수가 없을 때만 `submissions/`에 생성한다.
