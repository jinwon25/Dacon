# 노트북 안내

| 경로 | 역할 |
|---|---|
| `experiment_workbench.ipynb` | 현재 코드의 테스트·screen·학습·패키징·검증을 순서대로 호출하는 팀 공통 워크벤치 |
| `baseline/` | DACON 배포 RandomForest 학습·추론 예제 원본 |

새 실험 로직은 노트북 셀에 직접 쌓지 않고 `src/` 모듈과 `tests/`에 구현한다. 워크벤치는 실행 순서와 결과 확인 용도로만 사용한다.
