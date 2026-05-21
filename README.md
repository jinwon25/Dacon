# Dacon

데이콘(Dacon) 대회 참가 솔루션 모음. 각 폴더는 독립된 프로젝트이며, 자세한 내용은 폴더 안의 `README.md`를 참고하세요.

## 프로젝트

| 대회 | 상태 | 점수 (지표) | 핵심 접근 |
|---|---|---|---|
| [모기 비행 궤적 예측 AI 경진대회](./모기%20비행%20궤적%20예측%20AI%20경진대회) | 진행 중 | LB **0.6306** (R-Hit@1cm, 높을수록 좋음) | Physics candidate + Attn-GRU selector + boundary 보정 |
| [스마트 창고 출고 지연 예측 AI 경진대회](./스마트%20창고%20출고%20지연%20예측%20AI%20경진대회) | 완료 — **32위 (상위 10%)** | LB **9.86576** (MAE, 낮을수록 좋음) | 19-모델 mega-blend (GBDT 7 + Sequence NN 12, SLSQP) |

## 폴더 구조

```text
.
├── README.md                                # (이 파일) 레포 인덱스
├── .gitignore                               # 데이터/모델 산출물 일괄 제외
│
├── 모기 비행 궤적 예측 AI 경진대회/
│   ├── README.md                            # 솔루션 상세
│   ├── src/, scripts/, notebooks/           # 파이프라인 + 실험 코드
│   └── open/, outputs/, archive/            # gitignore (원본 데이터, 산출물)
│
└── 스마트 창고 출고 지연 예측 AI 경진대회/
    ├── README.md                            # 솔루션 상세
    ├── src/, notebooks/, docs/              # 학습/블렌드 코드 + 작업 로그
    └── data/, models/, submissions/         # gitignore (원본/체크포인트/제출)
```

## 공통 규칙

- **포함**: 소스 코드(`src/`, `scripts/`), 노트북, README, 작업 로그, `requirements.txt`
- **제외(.gitignore)**: 대회 원본 데이터, 모델 체크포인트, 생성 CSV/parquet, 캐시, `.venv/`, `.ipynb_checkpoints/`, `.env*`
- **Python**: 3.11 권장. 각 프로젝트의 `requirements.txt`로 의존성 격리

## 참여자

- GitHub: [@jinwon25](https://github.com/jinwon25)
