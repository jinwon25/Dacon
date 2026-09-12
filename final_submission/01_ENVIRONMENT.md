# 개발 및 실행 환경

## 로컬 검증 환경

Windows 11 Home, OS build 26200, x86_64, Intel i5-1340P, RAM 약 16 GB,
Python 3.11.2, PowerShell 5.1.
NumPy 2.2.6, pandas 2.2.3, SciPy 1.15.3, scikit-learn 1.6.1,
joblib 1.5.1, CatBoost 1.2.8, LightGBM 4.6.0, XGBoost 3.0.1,
torch 2.7.1, pytest 9.0.3으로 검증했다.
실제 설치 버전은 `python train.py preflight`가 출력한다.

## 학습 환경 구분

- 기본 목표 환경: `code/requirements.txt`.
  위 검증 환경과 달리 XGBoost는 3.2.0으로 고정돼 있다.
- H1: `code/h1/requirements_cat.txt`.
  CatBoost 1.2.10, scikit-learn 1.7.2, joblib 1.5.3, pandas 2.3.3, NumPy 2.2.6.
  별도 가상환경에서 학습한다. 모델은 depth 8, 1200 iterations, seed 42/43/44다.
  위 버전 조합의 별도 가상환경에서 전체 재학습해 원 트리와 일치함을 확인했다.
  해당 실행의 직접·간접 의존성은 `code/h1/requirements_verified.txt`에 고정했다.
- fallback XGBoost: XGBoost 3.2.0, `tree_method=hist`, `device=cuda:0`.
  전용 학습 코드, 114피처 계약과 고정 lookup을 함께 제공한다.
- 초기 V2 OOF, strict, 초기 부모 residual/퓨처스/legacy:
  Windows 11, Python 3.11.2, NumPy 2.2.6, pandas 2.2.3,
  LightGBM 4.6.0, scikit-learn 1.6.1, CatBoost 1.2.8.
  각 진입점은 공식 train 해시와 필요한 라이브러리 버전을 학습 전에 검사한다.
- workload H1과 C3 OOF: CatBoost 1.2.8, scikit-learn 1.6.1,
  NumPy 2.2.6, pandas 2.2.3, joblib 1.5.1인 기본 검증 환경에서 재학습했다.
- 추론: 리더보드에 제출한 `inference/requirements.txt`를 그대로 유지한다.

구성요소별 고정 버전이 다르므로 해당 requirements 파일에 맞춘 별도 가상환경을 사용한다.
버전이 다른 scikit-learn에서 저장 모델을 읽으면 경고가 발생할 수 있다.
fallback XGBoost 학습에는 CUDA를 지원하는 NVIDIA GPU 환경을 사용한다.
