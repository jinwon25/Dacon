# 모델링 전략

## 대회 구조 요약

BARAM 2026은 매일 09:00 KST 초기화 예보 중 다음 날 01:00부터 24시간 구간을 사용하는 day-ahead 발전량 예측 문제입니다. 테스트 기간에는 실제 발전량과 SCADA가 제공되지 않으므로, inference 입력은 공개된 LDAPS/GFS 예보와 달력 피처로 제한해야 합니다.

공식 평가 산식은 `0.5 * (1-NMAE) + 0.5 * FICR`입니다. 평가 대상은 실제 발전량이 그룹 설비용량의 10% 이상인 시간대이므로, 발전량이 충분히 나는 구간에서의 오차와 6%/8% 정산 경계 진입률이 중요합니다.

## 현재 채택한 접근

### 1. NWP 직접 예측 GBDT

LDAPS 16개 격자와 GFS 9개 격자의 바람 성분은 grid-level로 보존하고, 나머지 기상 변수는 공간 통계량으로 압축합니다. 각 타깃 그룹은 별도 모델로 학습합니다.

현재 기본 피처 수는 702개입니다. `src/features.py`에는 117m hub-height proxy와 터빈 위치 기반 inverse-distance weighted 피처도 구현되어 있지만, 전체 투입 시 2024 holdout 성능이 악화되어 기본 `--feature-set base`에서는 제외합니다.

### 2. 그룹별 모델 패밀리 선택

검증 결과:

| 타깃 | LightGBM best | CatBoost best | 선택 |
|---|---:|---:|---|
| `kpx_group_1` | 0.6669 | 검증 생략, 예비 실험에서 열세 | LightGBM |
| `kpx_group_2` | 0.6693 | 예비 실험 0.6682 | LightGBM |
| `kpx_group_3` | 0.6103 | 0.6251 | CatBoost |

CatBoost는 그룹 3처럼 label 기간이 짧고 발전량 분포가 다른 타깃에서 더 안정적이었습니다. 이는 ordered boosting이 작은 데이터와 분포 shift에서 LightGBM과 다른 bias/variance tradeoff를 만들기 때문으로 해석합니다.

### 3. 보정

각 모델의 raw prediction에 대해 2024 holdout에서 scale/offset grid search를 수행합니다. 이 보정은 MAE뿐 아니라 FICR 경계 진입률을 함께 개선하기 위한 장치입니다.

보정 리스크를 관리하기 위해 inference에서 `--calibration-strength`를 제공합니다.

- `1.0`: 검증 최적 보정 전체 적용
- `0.5`: public/private shift 대응용 절반 보정
- `0.0`: raw prediction

## 왜 deep SOTA를 바로 쓰지 않았는가

TFT, PatchTST, N-HiTS, TiDE는 모두 강한 시계열 예측 모델입니다. 다만 이 대회에서는 다음 제약 때문에 1차 제출 후보로는 GBDT가 더 실용적입니다.

- 테스트 기간 과거 발전량이 없어서 autoregressive target context를 자연스럽게 쓰기 어렵습니다.
- 학습 표본이 시간 단위 약 2.6만 행이고, 그룹 3은 2023-2024만 label이 있어 deep model 학습 데이터가 작습니다.
- 입력의 핵심 신호가 이미 미래 24시간 NWP covariate에 들어 있습니다.
- 2차 코드 검증에서 재현성과 누수 소명이 중요합니다.

따라서 SOTA 연구에서 가져올 부분은 모델 이름보다 구조적 아이디어입니다.

- TFT/TiDE: known future covariate 중심 설계
- PatchTST/N-HiTS: multi-horizon 예측과 scale decomposition 아이디어
- 풍력 power curve 연구: hub-height wind speed, wind speed cubed, turbine location weighting
- CatBoost: ordered boosting을 통한 prediction shift 완화

## 다음 실험 순서

1. **제출 우선순위**
   - `hybrid_lgbm_cat_g3_half_cal.csv`
   - `hybrid_lgbm_cat_g3_full_cal.csv`
   - `hybrid_lgbm_cat_g3_raw.csv`

2. **검증 확장**
   - 2024 전체 holdout 외에 월별 score/FICR breakdown 생성
   - public score와 local score 차이를 기록해 calibration strength 선택

3. **피처 실험**
   - `--feature-set own_idw_nohub`: 그룹별 위치 가중 피처 중 hub proxy 제외
   - `--feature-set own_idw`: 자기 그룹 위치 가중 전체 피처
   - 그룹 3에만 위치 가중 피처 적용

4. **앙상블**
   - 그룹 3 CatBoost raw와 LightGBM raw의 convex blending
   - seed ensemble은 feature/model 개선 이후 적용

5. **deep model 후보**
   - 하루 24시간 horizon을 하나의 sample로 재구성한 TiDE/TFT 스타일 MLP
   - 단, public score가 GBDT 대비 개선될 때만 유지

## 참고 자료

- DACON 대회 설명: https://dacon.io/competitions/official/236727/overview/description
- DACON 평가 방식: https://dacon.io/competitions/official/236727/overview/evaluation
- DACON 규칙: https://dacon.io/competitions/official/236727/overview/rules
- Temporal Fusion Transformers: https://arxiv.org/abs/1912.09363
- PatchTST: https://openreview.net/forum?id=Jbdc0vTOcol
- N-HiTS: https://arxiv.org/abs/2201.12886
- TiDE: https://arxiv.org/abs/2304.08424
- CatBoost: https://arxiv.org/abs/1706.09516
