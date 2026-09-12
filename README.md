# Dacon

데이콘(DACON) 및 연계 대회 참가 솔루션 모음. 각 폴더가 독립된 프로젝트이며,
상세한 접근과 재현 절차는 폴더 안의 `README.md`에 있습니다.

## 프로젝트

| 대회 | 상태 | 성적 (지표) | 핵심 접근 |
|---|---|---|---|
| [모기 비행 궤적 예측](./모기%20비행%20궤적%20예측%20AI%20경진대회) | **완료 · 2위 / 543팀** | Private **0.703151** (R-Hit@1cm, ↑) | 직교 메커니즘 앙상블 — Kalman 잔차 → Neural ODE → Frenet → 회전물리를 쌓아 상관 0.99 plateau 돌파 (0.6306 → 0.7032) |
| [투구 제구 성공 확률 예측](./투구%20제구%20성공%20확률%20예측%20AI%20온라인%20해커톤) | **완료 · 20위 / 1,090팀** (LG Aimers 9기, 5인 팀) | Public **1182.9497** (Brier Skill Score, ↑) | 3-도메인 라우팅 + 부모 보존 국소 보정 — 검증된 부모를 수치적으로 보존한 채 한 도메인에만 신호 주입 (749.52 → 1182.95) |
| [스마트 창고 출고 지연 예측](./스마트%20창고%20출고%20지연%20예측%20AI%20경진대회) | **완료 · 32위 / 607팀** (상위 5.3%) | LB **9.86576** (MAE, ↓) | 19-모델 mega-blend (GBDT 7 + Sequence NN 12, SLSQP 가중 최적화) |
| [제3회 풍력발전량 예측 (BARAM 2026)](./제3회%20풍력발전량%20예측%20AI%20경진대회%20-%20BARAM%202026) | 진행 중 · 291위 / 985팀 | Public **0.6474704399** (0.5·(1-NMAE) + 0.5·FiCR, ↑) | LDAPS/GFS 공간 바람장·물리 피처 LightGBM + 그룹별 public-positive 가중. 승격 게이트를 q05 하한 기준으로 강화 |
| [식음업장 메뉴 수요 예측](./식음업장%20메뉴%20수요%20예측%20AI%20온라인%20해커톤) | 진행 중 (LG Aimers) | Private **0.5481** (가중 SMAPE, ↓) | 0 제외 SMAPE 특성을 노린 업장별 nz-mean 무학습 블렌드 (0.694 → 0.548). LSTM·단일 GBDT는 열세 |
| [K리그 최종 패스 좌표 예측](./K리그-서울시립대%20공개%20AI%20경진대회) | 학습 목적 (종료 대회) | 로컬 CV **13.95** (유클리드 거리, ↓) | SQL(DuckDB) 기반 EDA → GBDT + 시퀀스 GRU 앙상블. 관찰 → 가설 → 검증 사슬 기록에 중점 |
| [내가 축구 감독이라면](./내가%20축구%20감독이라면) | 완료 (월간 해커톤 출품) | 웹서비스 · [데모 ↗](https://jinwon25.github.io/Dacon/) | 월드컵 전술 개입 워크스페이스 — 포메이션·압박·교체 조정의 이점과 리스크를 비교 |

↑ = 높을수록 좋음, ↓ = 낮을수록 좋음. 순위는 DACON 최종 리더보드 기준입니다.

## 폴더 구조

```text
.
├── README.md                                # (이 파일) 레포 인덱스
├── .gitignore                               # 데이터·모델 산출물 일괄 제외
│
├── 모기 비행 궤적 예측 AI 경진대회/
│   ├── README.md                            # 솔루션 상세
│   ├── src/, notebooks/, docs/              # 학습·블렌드 코드 + 솔루션 문서·작업 로그
│   ├── submissions/                         # 최종 제출 + 재현 패키지 (rebuild.py, inputs/)
│   └── data/                                # gitignore (원본 데이터, 캐시)
│
├── 스마트 창고 출고 지연 예측 AI 경진대회/
│   ├── README.md                            # 솔루션 상세
│   ├── src/, notebooks/, docs/              # 학습·블렌드 코드 + 작업 로그
│   ├── submissions/                         # 최종 제출 (submission.csv)
│   └── data/, models/                       # gitignore (원본·체크포인트)
│
├── 제3회 풍력발전량 예측 AI 경진대회 - BARAM 2026/
│   ├── README.md                            # 솔루션 상세 + Public 제출 기록
│   ├── train.py, inference.py, src/, docs/  # 학습·추론 분리 + 실험 보고서
│   └── data/, artifacts/, submissions/      # gitignore (원본·모델·생성 제출)
│
├── 식음업장 메뉴 수요 예측 AI 온라인 해커톤/
│   ├── README.md                            # 솔루션 상세
│   ├── src/, notebooks/, docs/              # 베이스라인·블렌드·지표 코드 + 작업 로그
│   └── data/, submissions/                  # gitignore (원본·생성 제출)
│
├── K리그-서울시립대 공개 AI 경진대회/
│   ├── README.md                            # 솔루션 상세 + 학습 기록
│   ├── sql/, src/, docs/                    # DuckDB EDA + 모델링 + 작업 로그
│   └── data/, submissions/                  # gitignore (원본·생성 제출)
│
├── 투구 제구 성공 확률 예측 AI 온라인 해커톤/   # 5인 팀, 2,378 파일
│   ├── README.md                            # 솔루션 상세 + 계보
│   ├── final_submission/                    # Private Score 재현 패키지 (v345)
│   ├── src/champion, core, archive/         # 챔피언 계보 / 공통 / 종료된 실험
│   ├── research/reports, configs/           # 실험 기록과 제출 원장
│   └── data/, artifacts/, submissions/      # gitignore (대회 데이터 파생물)
│
└── 내가 축구 감독이라면/                      # 전술 웹서비스 (GitHub Pages로 배포)
    ├── README.md, docs/                     # 기획·설계 문서
    ├── src/, tests/                          # React + TypeScript, Playwright e2e
    └── node_modules/, dist/                  # gitignore (의존성·빌드 산출물)
```

투구 제구 대회는 대회 제공 데이터에서 파생된 산출물(OOF 번들, 제출 ZIP, 선수 연결표)이
재배포 금지 대상이라 저장소에 올리지 않습니다. 어떤 파일이 왜 제외되는지는
[해당 프로젝트의 정책](./투구%20제구%20성공%20확률%20예측%20AI%20온라인%20해커톤/README.md)에 정리돼 있습니다.

## 공통 규칙

- **포함**: 소스 코드(`src/`), 노트북, README, 작업 로그, `requirements.txt`, 최종 제출 파일
- **제외(.gitignore)**: 대회 원본 데이터, 모델 체크포인트, 생성 CSV/parquet, 캐시, `.venv*`, `.env*`
- **Python**: 3.11 기준. 각 프로젝트의 `requirements.txt`로 의존성 격리

## 참여자

- GitHub: [@jinwon25](https://github.com/jinwon25)
