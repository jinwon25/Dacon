# Dacon

데이콘(Dacon) 대회 참가 솔루션 모음. 각 폴더는 독립된 프로젝트이며, 자세한 내용은 폴더 안의 `README.md`를 참고하세요.

## 프로젝트

| 대회 | 상태 | 점수 (지표) | 핵심 접근 |
|---|---|---|---|
| [모기 비행 궤적 예측 AI 경진대회](./모기%20비행%20궤적%20예측%20AI%20경진대회) | 완료 — **Private 2위** | LB Private **0.703151** (R-Hit@1cm, 높을수록 좋음) | 직교 메커니즘 앙상블 — Kalman 잔차 → Neural ODE → Frenet → CREE 회전물리 base를 쌓고 수동 α 주입으로 corr~0.99 plateau 돌파 (베이스라인 0.6306 → 0.703) |
| [스마트 창고 출고 지연 예측 AI 경진대회](./스마트%20창고%20출고%20지연%20예측%20AI%20경진대회) | 완료 — **32위 (상위 10%)** | LB **9.86576** (MAE, 낮을수록 좋음) | 19-모델 mega-blend (GBDT 7 + Sequence NN 12, SLSQP) |
| [식음업장 메뉴 수요 예측 AI 온라인 해커톤](./식음업장%20메뉴%20수요%20예측%20AI%20온라인%20해커톤) | 연구 중 | LB Private **0.5481** (가중 SMAPE, 낮을수록 좋음) | 업장별 nz-mean 블렌드 — 0 제외 SMAPE 특성 활용(베이스라인 0.694 → 0.548). LSTM·단일 GBDT는 열세 |
| [제3회 풍력발전량 예측 AI 경진대회 - BARAM 2026](./제3회%20풍력발전량%20예측%20AI%20경진대회%20-%20BARAM%202026) | 연구 중 | LB Public **0.635279** (0.5·1-NMAE + 0.5·FICR, 높을수록 좋음) | LDAPS/GFS 공간 바람장·물리 피처 LightGBM, 2024 시간순 검증 |
| [투구 제구 성공 확률 예측 AI 온라인 해커톤](https://github.com/jinwon25/lg-aimers9-pitch-control) ↗ | 완료 (LG Aimers 9기, 3인 팀) | LB Public **1182.9497** (Brier Skill Score, 높을수록 좋음) | 3-도메인 라우팅 + 부모 보존 국소 보정 — 강한 부모를 수치적으로 보존한 채 한 도메인에만 검증된 신호를 주입 (첫 제출 749.52 → 1182.95) |
| [K리그 경기 내 최종 패스 좌표 예측 AI 경진대회](./K리그-서울시립대%20공개%20AI%20경진대회) | 학습 목적 (종료된 대회) | 로컬 CV **13.95** (유클리드 거리, 낮을수록 좋음) | SQL(DuckDB) 기반 EDA → GBDT + 시퀀스 GRU 앙상블. 관찰→가설→검증 사슬 기록에 중점 |
| [내가 축구 감독이라면](./내가%20축구%20감독이라면) | 완료 (월간 해커톤 출품) | 웹서비스 (점수 지표 없음) | 월드컵 전술 개입 워크스페이스 — 포메이션·압박·교체 조정의 이점/리스크 비교 |

## 폴더 구조

```text
.
├── README.md                                # (이 파일) 레포 인덱스
├── .gitignore                               # 데이터/모델 산출물 일괄 제외
│
├── 모기 비행 궤적 예측 AI 경진대회/
│   ├── README.md                            # 솔루션 상세
│   ├── src/, notebooks/, docs/              # 학습/블렌드 코드 + 솔루션 문서·작업 로그·리포트
│   ├── submissions/                         # 최종 제출 + 재현 패키지 (rebuild.py, inputs/)
│   └── data/                                # gitignore (원본 데이터, 캐시)
│
├── 스마트 창고 출고 지연 예측 AI 경진대회/
│   ├── README.md                            # 솔루션 상세
│   ├── src/, notebooks/, docs/              # 학습/블렌드 코드 + 작업 로그
│   ├── submissions/                         # 최종 제출 (submission.csv)
│   └── data/, models/                       # gitignore (원본/체크포인트)
│
├── 식음업장 메뉴 수요 예측 AI 온라인 해커톤/
    ├── README.md                            # 솔루션 상세
    ├── src/, notebooks/, docs/              # 베이스라인/블렌드/지표 코드 + 작업 로그
    └── data/, submissions/                  # gitignore (원본/생성 제출)
│
├── 제3회 풍력발전량 예측 AI 경진대회 - BARAM 2026/
│   ├── README.md                            # 솔루션 상세 + Public 제출 기록
│   ├── train.py, inference.py, src/, docs/  # 학습·추론 분리 + 문헌 전략
│   └── data/, artifacts/, submissions/      # gitignore (원본/모델/생성 제출)
│
├── K리그-서울시립대 공개 AI 경진대회/
│   ├── README.md                            # 솔루션 상세 + 학습 기록
│   ├── sql/, src/, docs/                    # DuckDB EDA + 모델링 + 작업 로그
│   └── data/, submissions/                  # gitignore (원본/생성 제출)
│
├── 내가 축구 감독이라면/                      # 별도 저장소 (웹서비스, GitHub Pages 배포)
│
└── 투구 제구 성공 확률 예측 AI 온라인 해커톤/  # 별도 저장소로 분리
    └── github.com/jinwon25/lg-aimers9-pitch-control
```

투구 제구 대회는 규모(2,378 파일)와 팀 협업 이력 때문에 별도 저장소로 분리했다.
대회 제공 데이터에서 파생된 산출물(OOF·제출 ZIP·lookup)은 어느 저장소에도 올리지 않는다.

## 공통 규칙

- **포함**: 소스 코드(`src/`), 노트북, README, 작업 로그, `requirements.txt`, 최종 제출 파일(`submissions/`)
- **제외(.gitignore)**: 대회 원본 데이터, 모델 체크포인트, 생성 CSV/parquet, 캐시, `.venv/`, `.ipynb_checkpoints/`, `.env*`
- **Python**: 3.11 권장. 각 프로젝트의 `requirements.txt`로 의존성 격리

## 참여자

- GitHub: [@jinwon25](https://github.com/jinwon25)
