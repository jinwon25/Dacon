# 모기 비행 궤적 예측 AI 경진대회

데이콘에서 주최한 **모기 비행 궤적 예측 AI 경진대회** 솔루션 정리 문서입니다. 3D LiDAR 기반 관측 궤적을 입력으로 받아, 현재 시점 이후의 모기 위치를 예측하는 문제를 다룹니다.

> 진행 중 (대회 종료 전, 상위권 0.685+ 추격 중).

## 결과 요약

| 항목 | 값 |
|---|---:|
| 현재 best LB | **0.6306** |
| 주요 로컬 검증 지표 | OOF R-Hit@1cm 0.6424 |
| 대표 제출 버전 | v10 |
| 최종 실험 방향 | Physics Ladder Candidate Selection |

평가 지표는 **R-Hit@1cm**입니다. 예측 좌표가 실제 좌표와 1cm 이내에 들어오면 hit로 계산되며, 점수가 높을수록 좋습니다. 입력 궤적은 40ms 간격의 과거 관측값이고, 목표는 마지막 관측 이후 약 **80ms** 시점의 `x, y, z` 좌표입니다.

## 문제 정의

- 입력: `open/train/*.csv`, `open/test/*.csv`의 시계열 3D 좌표 (`timestep_ms`, `x`, `y`, `z`)
- 정답: `open/train_labels.csv`의 미래 좌표 (`x`, `y`, `z`)
- 제출: `sample_submission.csv` 형식의 test별 예측 좌표
- 데이터 규모: train 10,000개, test 10,000개 CSV

원본 대회 데이터는 `open/` 아래에 배치하지만, 대회 제공 데이터이므로 Git에는 포함하지 않습니다.

## 접근 방식

이 솔루션은 좌표를 직접 회귀하는 대신, 물리적으로 가능한 후보 좌표군을 만든 뒤 모델이 후보를 선택하고 1cm 경계 근처에서만 작은 보정을 수행하는 구조입니다.

1. **Physics candidates**
   - 마지막 관측점, 속도, 가속도, Frenet frame, turn, jerk, latency 계열 후보를 생성합니다.
   - 직접적인 환경 식별 대신 관측 차이를 후보 선택 문제로 변환합니다.

2. **Attn-GRU selector**
   - 최근 궤적 요약과 후보별 feature를 결합해 후보 score를 예측합니다.
   - prior, pairwise loss, distillation을 사용해 특정 노이즈 패턴에 과적합되는 것을 줄였습니다.

3. **Tiny boundary correction**
   - 후보 선택 후 1cm hit boundary 근처 샘플만 작은 residual로 보정합니다.
   - 보정량은 cap으로 제한해 후보 물리를 크게 망가뜨리지 않도록 했습니다.

4. **Ensemble / hill climbing**
   - v10, v13, v16 계열 실험 결과를 조합하며 OOF 기준으로 ensemble 후보를 비교했습니다.

## 폴더 구조

```text
.
├── README.md                     # 프로젝트 요약, 성과, 재현 방법
├── requirements.txt              # 최소 재현용 Python 의존성
├── .gitignore
│
├── src/
│   └── pipeline.py               # 현재 대표 파이프라인 export
│
├── notebooks/
│   ├── pipeline.ipynb            # 같은 흐름의 노트북
│   └── reference.ipynb           # 참고 실험 노트북
│
├── open/                         # 대회 원본 데이터, gitignore
│   ├── train/
│   ├── test/
│   ├── train_labels.csv
│   └── sample_submission.csv
│
├── archive/                      # 과거 모델/실험 보관, gitignore
├── outputs/                      # 실행 산출물, gitignore
└── run_log.json                  # 로컬 실험 로그, gitignore
```

대표 실행 코드는 `src/pipeline.py`로 분리했고, 과거 v5-v8 실험 폴더는 ignored `archive/legacy_versions/`에 로컬 보관했습니다. 장기적으로는 과거 실험을 Git tag(`v10`, `v13`, `v16`)로 대체하면 더 표준적인 구조가 됩니다.

## 재현 방법

Python 3.11 환경을 권장합니다.

```bash
pip install -r requirements.txt
```

대회에서 받은 데이터를 아래처럼 배치합니다.

```text
open/
├── train/
├── test/
├── train_labels.csv
└── sample_submission.csv
```

현재 대표 파이프라인은 `src/pipeline.py`에 있습니다. 해당 파일은 노트북 export 형태라 selector 학습과 boundary 보정 예제가 하단에 함께 들어 있습니다.

```bash
python src/pipeline.py --smoke-check
python src/pipeline.py --run-selector
```

실행 결과는 `outputs/` 아래에 생성되며, 제출 파일과 score bank는 Git에 포함하지 않습니다.

## 커밋 정책

- 포함: README, `.gitignore`, `src/pipeline.py`, 출력이 정리된 노트북
- 제외: `open/` 원본 데이터, `.npz` 모델 state, 제출 CSV, 로컬 실행 로그, 캐시/출력 디렉터리
- 노트북은 커밋 전 출력 셀을 비우는 것을 권장합니다.

## 환경

- Python: 3.11 권장
- 학습: Colab T4 (원격 커널) + 로컬 (Windows) 병행
- 핵심 의존성: PyTorch 2.7+, NumPy 2.x (`requirements.txt` 참고)

## 남은 정리 작업

- `archive/legacy_versions/` 실험 폴더를 Git tag 기반 구조로 전환
- `pipeline.ipynb` 출력 셀 제거 또는 `.py` 중심 재현 경로 확정
- 제출별 LB 기록을 `WORK_LOG.md`로 분리
- 최소 의존성을 `requirements.txt`로 고정
