# 제3회 풍력발전량 예측 AI 경진대회 (BARAM 2026)

데이콘에서 주최한 [**제3회 풍력발전량 예측 AI 경진대회**](https://dacon.io/competitions/official/236727/overview/description) 솔루션.

## 최종 결과

> **최종 291위 / 985팀** · **Public 0.6474704399** (제출 `1508386`)
> 초기 블렌드(0.6367) 대비 **+0.0108**.

| 단계 | Public | 돌파 내용 |
|---|---:|---|
| 초기 블렌드 | 0.6367 | LightGBM/CatBoost 후보 블렌드 |
| `blend_v1` | 0.6395 | 블렌드 구성 정리 |
| SCADA proxy 5% 주입 | 0.6403 | 전역 최적이 5% 부근으로 좁음을 확인 |
| cross-group 25%/6% 게이트 | 0.6415 | 그룹2↔3 정규화 상관(2023 0.914, 2024 0.945) 활용 |
| settlement-aware meta-gate | 0.6417 | 정산 구간을 인지한 게이트 |
| **그룹별 public-positive 가중** | **0.6475** | **현재 최고** |

평가 지표: `0.5 × (1 - NMAE) + 0.5 × FiCR` (높을수록 좋음). **실제 발전량이 설비용량의
10% 이상인 시점만** 평가 대상이며, FiCR은 시간별 정산을 실제 발전량으로 가중한다.

---

## 문제 정의

- **입력**: LDAPS/GFS 일전(day-ahead) 수치예보(NWP) — 풍속·풍향·기온·기압 등 격자 예보
- **출력**: 2025년 **시간별** 풍력발전량, KPX 3개 그룹 각각
- **평가**: 그룹별 설비용량 정규화 MAE와 6%/8% 정산 구간 적중률(FiCR)의 50:50 평균
- **핵심 성질**: 정산 구간이 **계단형**이라 평균 오차를 줄이는 것과 구간 적중을 높이는 것이
  서로 다른 방향을 요구한다. 실제로 1-NMAE는 오르는데 FiCR이 떨어져 총점이 내려간
  제출이 여러 번 나왔다.

---

## 솔루션 아키텍처

단일 모델이 아니라 **검증된 부모 블렌드에 좁은 보정을 얹는** 구조다.

```text
LightGBM/CatBoost 후보 블렌드 (base)
  + SCADA proxy stack 5%              # 전역 최적이 좁아 그룹별로 전환
  + cross-group 25% / 6% 불일치 게이트  # 그룹2↔3 안정 상관 활용
  + settlement-aware meta-gate         # 정산 구간 인지 보정
  + 그룹별 public-positive 가중         # g1 1.375 / g2 1.825 / g3 동결
```

### 핵심 인사이트 — 병목은 모델이 아니라 승격 거버넌스였다

지표 구현을 공식 노트북과 대조한 결과 **로컬 산술은 정확히 일치**했다. 문제는 후보를
승격시키는 규칙이었다. 전체 연도 기준으로 양수인 후보가, 공개 채점과 같은 크기(40%)의
하위 꼬리와 그 여집합(60%)에서는 **음수**인 채로 게이트를 통과하고 있었다.

제출 `1504383`의 경우 1만 회 국소 분할에서 관측된 공개 쌍 점수 차이가 음수 구간에
놓였고, 시뮬레이션한 공개 q05는 이미 음수였다. 이후 정책
`baram-public-v4-subset-safe`는 **q05 하한이 음수가 아닐 것**을 요구하도록 바꿨다.

### 시도했지만 효과 없음

| 시도 | 결과 |
|---|---|
| turbine 기반 group-3 주입 (양·음 방향 모두) | 두 방향 다 Public 하락, 계열 종료 |
| FiCR 분포 레이어 | 자체 분위 베이스라인은 개선했으나 상위 앙상블로 전이 실패 |
| phase/regime 교차 앙상블 | H2 exact-OOF 이득이 Public으로 전이 실패(FiCR −0.0012) |
| spatial-temporal graph multitask | 로컬 H2 개선, Public에서 FiCR 하락으로 기각 |
| power-curve residual 선택 주입 | 0.6401로 하락 |
| SCADA 상태 분류기 MoE, 9-threshold ordinal | AUC 0.653에 그치고 모든 proxy 점수 하락 |

---

## 폴더 구조

```text
train.py, inference.py      학습·추론 진입점
src/features.py             NWP 피처 엔지니어링
src/metrics.py              공식 지표(1-NMAE/FiCR) 로컬 구현
experiments/                실험 스크립트 (블렌드·OOF·게이트·외부데이터)
agent_service/              승격 정책·규정 준수 검사
docs/reports/               실험별 보고서와 판정 근거
tests/                      지표·정책·외부데이터 규정 테스트 (373 passed)
data/, artifacts/, submissions/   gitignore
```

## 재현 방법

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python -m pytest -q        # 데이터 없이 373개 통과
```

공식 데이터를 `data/`에 배치한 뒤 `train.py` → `inference.py` 순서로 실행한다.
자세한 명령은 아래 **Main Commands**를 참고한다.

## 대회 규정 준수

- 평가 기간의 실제 발전량과 SCADA는 사용하지 않는다. SCADA는 학습 기간 proxy 모델링에만 쓴다.
- 외부 데이터는 예측 시점 공개 여부를 만족해야 하며, 출처 manifest·체크섬·라이선스를 함께 기록한다.
- 사후 취득 가능한 Open-Meteo 과거/이전 실행 데이터는 공개 시점 증거 없이는 제출에 쓰지 않는다.
- 원격 추론 API를 사용하지 않는다.

이 규칙은 `agent_service/compliance.py`가 강제하고 `tests/test_external_data_compliance.py`가 검증한다.

---


상세한 실험 진행 기록은 [연구 이력](docs/RESEARCH_HISTORY.md)으로 분리했습니다.
