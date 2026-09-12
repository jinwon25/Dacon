# H1

CatBoost 3시드 앙상블과 affine 보정으로 구성된다.
`rf.pkl`은 저장 파일명이며 모델 종류는 RandomForest가 아니다.
depth 8, 1200 iterations, seed 42/43/44, 고정 center 0.5854452601930041,
alpha 1.09를 사용한다. 피처는 82개다.

`code/`에서 H1 전용 환경으로 다음 명령을 실행한다.

```powershell
python train.py train-h1 --data-dir data --output-dir artifacts/h1_01
```

환경은 `requirements_cat.txt`, 설정은 `training_recipe.json`을 따른다.
`exp/build_asof.py`가 학습을 담당하고 `final_train.py`는 공통 피처 함수를 제공한다.
기존 모델 설정 대조와 세 모델의 전체 신규 학습 동일성 검증은 구분한다.
공식 평가 지표는 Brier Skill Score이며 상관계수 제곱 지표가 아니다.
