# 노트북 안내

| 경로 | 역할 |
|---|---|
| `v26_champion_reproduction.ipynb` | 현재 champion `submit_v26.zip`을 그대로 실행·검증하고 동일 recipe로 재빌드하는 단일 진입점. 팀원이 브랜치를 받아 가장 먼저 여는 노트북 (`README.md`의 과거 v27 champion 표기는 정정 대상이니 이 노트북과 `reports/submissions.csv`를 우선한다) |
| `experiment_workbench.ipynb` | 과거 v14→v19 pressure EB 계열 테스트·screen·학습·패키징·검증을 순서대로 호출하던 워크벤치 (연구 이력 참고용, v22 이후 R_ANCHOR 계열은 다루지 않음) |
| `baseline/` | DACON 배포 RandomForest 학습·추론 예제 원본 |

새 실험 로직은 노트북 셀에 직접 쌓지 않고 `src/` 모듈과 `tests/`에 구현한다. 워크벤치는 실행 순서와 결과 확인 용도로만 사용한다.
