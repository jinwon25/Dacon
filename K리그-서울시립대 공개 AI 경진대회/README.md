# K리그 경기 내 최종 패스 좌표 예측 AI 경진대회

데이콘에서 주최한 [**K리그 경기 내 최종 패스 좌표 예측 AI 경진대회**](https://dacon.io/competitions/official/236647/overview/description) 솔루션 (서울시립대·한국프로축구연맹 주최).

> ℹ️ **종료된 대회를 학습 목적으로 진행**한 기록입니다. 데이터 작업은 **SQL(DuckDB)** 로 하고,
> 모든 의사결정을 **관찰 → 가설 → 로컬검증 → 배움**의 사슬로 남기는 데 중점을 뒀습니다.
> 자세한 진행 과정과 교훈은 [docs/WORK_LOG.md](./docs/WORK_LOG.md) 참고.

## 최종 결과 (진행 중)

| 단계 | 방법 | 로컬 CV | 리더보드 |
|---|---|---:|---:|
| 베이스라인 | 도착 = 출발 | 20.34 | 20.7314 |
| 베이스라인 | 출발 + 평균전진 + clip | 18.16 | 18.3662 |
| GBDT v1 | LightGBM (18 피처) | 14.66 | 14.7661 |
| GBDT v3 | + 각도 + 타깃인코딩 | 14.39 | 14.5870 |
| GBDT v4/v5 | + 풍부 피처(같은팀터치/이벤트종류/템포) | 14.01 | — |
| **앙상블** | GBDT + 강화 GRU(시퀀스) | **13.95** | — |

평가 지표: 마지막 패스 도착 좌표 예측의 **유클리드 거리**(낮을수록 좋음). 로컬 CV가 리더보드를 잘 추종(격차 ~0.2).

---

## 문제 정의

- **에피소드** = 공이 라인 밖으로 나갈 때까지 이어진 플레이 한 묶음(이벤트 시퀀스).
- **목표** = 각 에피소드의 **마지막 패스가 도착하는 좌표 `(end_x, end_y)`** 예측.
- 테스트는 마지막 패스의 **출발점은 주어지고 도착점만 가려져** 있음.
- **좌표계**: 105 × 68 그리드(피파 권장 규격), 모든 데이터 "왼쪽→오른쪽 공격" 통일. 중앙 = (52.5, 34).
- **데이터**: `train.csv`(356,721 이벤트 / 198 경기 / 15,435 에피소드), `test/`(2,414 에피소드), `match_info.csv`, `sample_submission.csv`.
  - 이벤트 컬럼: `game_id, period_id, episode_id, time_seconds, team_id, player_id, action_id, type_name, result_name, start_x/y, end_x/y, is_home, game_episode`.

---

## 솔루션 아키텍처

핵심 인사이트(상위 공유코드와 일치): **"답은 마지막 몇 개 이벤트의 위치·방향·거리·타이밍이 거의 결정한다."** → 긴 시퀀스 통째 이해보다 **요약 피처 + GBDT**가 강하고, 마지막 push는 **시야가 다른 시퀀스 모델 앙상블**.

```
1) EDA (SQL/DuckDB)  →  데이터 이해, 베이스라인 직관(평균 전진 +13.5)
2) 로컬 검증 구축      →  game_id 5-fold, 제출 없이 평균 유클리드 거리 측정
3) GBDT (LightGBM)    →  에피소드 요약 피처 + 변위(도착-출발) 타깃 + OOF 타깃인코딩
4) 앙상블             →  GBDT + 강화 GRU(시퀀스+스칼라+시드배깅), 최적 가중 블렌드
```

### 핵심 설계 결정

1. **변위 타깃**: 절대 좌표 대신 `도착-출발`을 예측(잔차 학습) → 안정적, -0.21.
2. **OOF 타깃 인코딩**: `player_id`를 그룹 평균 좌표로. 누수 방지 위해 자기 fold 제외하고 계산.
3. **풍부한 피처**: 같은 팀 직전 터치, 이벤트 종류 흐름, 다중 윈도우(3/5/10), sin/cos 각도, **템포(t_gap)**.
4. **decorrelated 앙상블**: 같은 피처 GBDT끼린 corr ~0.99로 무의미 → 시퀀스 GRU로 다른 시야를 더해 14.0 → 13.95.

### 시도했지만 효과 없음 / 배운 것

| 시도 | 결과 / 교훈 |
|---|---|
| 직전 이동(모멘텀) 단독 예측 | A보다 나쁨 — 단독 무용 피처도 *맥락 속*에선 강력(importance 1·2위) |
| LGB/CatBoost/XGB 앙상블(같은 피처) | OOF corr ~0.99 → 이득 +0.045뿐 |
| 편의로 `prev_type` 제거 | +0.8 악화 — 직전 이벤트 종류가 강한 피처 |
| 템포 피처 추가(v5) | 정체 — GBDT 포화 |

---

## 폴더 구조

```text
.
├── README.md
├── .gitignore
├── sql/                      # DuckDB EDA 쿼리 (01_overview.sql)
├── src/
│   ├── run_sql.py            # .sql 실행기
│   ├── local_cv.py           # 로컬 검증(제출 없이 점수)
│   ├── features.py / features_v4.py   # 에피소드 → 피처 (SQL)
│   ├── make_baseline.py / make_submission.py
│   ├── train_lgbm.py / exp_target.py / train_v3.py / train_v4.py / train_v5.py
│   ├── train_gru.py / train_gru2.py   # 시퀀스 모델
│   ├── ensemble_final.py / multi_blend.py
│   └── submit.py             # DACON 제출 API 헬퍼
├── docs/WORK_LOG.md          # 진행 과정·교훈 (학습 핵심)
├── submissions/              # 생성 제출 (gitignore, *.csv)
└── data/                     # 대회 원본 + 캐시 (gitignore)
```

---

## 재현 방법

Python 3.11. CPU만으로 가능(GRU 학습 포함).

```bash
pip install -r requirements.txt          # duckdb, lightgbm, torch 등
# 데이터를 data/ 에 배치 (train.csv, test/, sample_submission.csv ...)

python src/run_sql.py sql/01_overview.sql # EDA
python src/local_cv.py                    # 로컬 검증 + 단순 베이스라인
python src/train_v5.py                    # GBDT (피처 → 변위 LightGBM)
python src/train_gru2.py                  # 강화 GRU (시퀀스 + 스칼라 + 배깅)
python src/ensemble_final.py              # GBDT + GRU 블렌드 → 제출 파일
```

제출(DACON API, 선택):
```bash
# .env 에 DACON_TOKEN, DACON_TEAM 설정(절대 커밋 금지) 후
python src/submit.py submissions/<파일>.csv "메모"
```

---

## 대회 규칙 준수

- **외부 데이터 미사용**: 제공 데이터만 사용. ✓
- **에피소드 독립 추론**: 각 예측은 해당 episode 내부 시퀀스만 입력. 다른 에피소드 정보 미사용. ✓
- **test 누수 금지**: 학습은 train만. 타깃 인코딩은 OOF(자기 fold 제외)로 누수 차단. ✓
- **원격 API 모델 미사용**: 모두 로컬 실행(LightGBM, PyTorch). ✓

## 사용 라이브러리

| 라이브러리 | 용도 |
|---|---|
| DuckDB | CSV에 직접 SQL (EDA·피처) |
| LightGBM / CatBoost / XGBoost | GBDT 회귀 |
| PyTorch | GRU 시퀀스 모델 |
| pandas / numpy / scipy | 데이터·블렌드 |

## 환경

- Python 3.11, CPU 전용(GPU 불필요). OS: Windows 11.
