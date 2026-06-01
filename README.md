# Dacon

데이콘(Dacon) 대회 참가 솔루션 모음. 각 폴더는 독립된 프로젝트이며, 자세한 내용은 폴더 안의 `README.md`를 참고하세요.

## 프로젝트

| 대회 | 상태 | 점수 (지표) | 핵심 접근 |
|---|---|---|---|
| [모기 비행 궤적 예측 AI 경진대회](./모기%20비행%20궤적%20예측%20AI%20경진대회) | 완료 — **Private 2위** | LB Private **0.703151** (R-Hit@1cm, 높을수록 좋음) | 직교 메커니즘 앙상블 — Kalman 잔차 → Neural ODE → Frenet → CREE 회전물리 base를 쌓고 수동 α 주입으로 corr~0.99 plateau 돌파 (베이스라인 0.6306 → 0.703) |
| [스마트 창고 출고 지연 예측 AI 경진대회](./스마트%20창고%20출고%20지연%20예측%20AI%20경진대회) | 완료 — **32위 (상위 10%)** | LB **9.86576** (MAE, 낮을수록 좋음) | 19-모델 mega-blend (GBDT 7 + Sequence NN 12, SLSQP) |
| [식음업장 메뉴 수요 예측 AI 온라인 해커톤](./식음업장%20메뉴%20수요%20예측%20AI%20온라인%20해커톤) | 연구 중 | LB Private **0.5481** (가중 SMAPE, 낮을수록 좋음) | 업장별 nz-mean 블렌드 — 0 제외 SMAPE 특성 활용(베이스라인 0.694 → 0.548). LSTM·단일 GBDT는 열세 |

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
│   └── data/, models/, submissions/         # gitignore (원본/체크포인트/제출)
│
└── 식음업장 메뉴 수요 예측 AI 온라인 해커톤/
    ├── README.md                            # 솔루션 상세
    ├── src/, notebooks/, docs/              # 베이스라인/지표 코드 + 작업 로그
    └── data/, submissions/                  # gitignore (원본/제출)
```

## 공통 규칙

- **포함**: 소스 코드(`src/`), 노트북, README, 작업 로그, `requirements.txt`, 최종 제출 파일(`submissions/`)
- **제외(.gitignore)**: 대회 원본 데이터, 모델 체크포인트, 생성 CSV/parquet, 캐시, `.venv/`, `.ipynb_checkpoints/`, `.env*`
- **Python**: 3.11 권장. 각 프로젝트의 `requirements.txt`로 의존성 격리

## 참여자

- GitHub: [@jinwon25](https://github.com/jinwon25)
