# 협업 가이드

이 문서는 세 팀원이 서로의 작업을 안전하게 검토하고, 같은 실험을 반복하지 않도록 하기 위한 공통 규칙입니다.

## 브랜치를 만들어 작업해 주세요

`main`에서 직접 파일을 수정하거나 push하지 마세요. 작업을 시작할 때 최신 `main`을 받은 뒤 새로운 브랜치를 만들어 주세요.

```powershell
git switch main
git pull --ff-only
git switch -c exp/core-reliability-router
```

브랜치 이름은 다음 형식을 권장합니다.

```text
exp/anchor-month-gate
exp/core-reliability-router
exp/f-exact-asof-v2
fix/inference-parity
docs/experiment-protocol
```

## 한 Pull Request에는 한 가지 목적만 담아 주세요

모델 가설과 문서 정리, 버그 수정처럼 성격이 다른 작업은 서로 다른 Pull Request로 나눠 주세요. 변경 범위가 작을수록 팀원이 검토하고 문제가 생겼을 때 되돌리기 쉽습니다.

병합 전에는 다음 사항을 확인해 주세요.

1. 최소 한 명의 팀원에게 리뷰를 받아 주세요.
2. GitHub Actions의 `ci`가 성공했는지 확인해 주세요.
3. 가능하면 squash merge를 사용해 주세요.
4. 병합 후 작업 브랜치를 삭제해 주세요.

## 모델 실험 결과를 남겨 주세요

모델 변경 Pull Request에는 다음 내용을 작성해 주세요.

1. 검증하려는 가설과 야구·통계적 근거
2. 학습 시즌과 검증 시즌
3. 부모 기준선과 후보의 전체 BSS 차이
4. `R_CORE`, `R_ANCHOR`, `F`별 결과
5. 월별 최악값과 bootstrap 하위 5% 개선
6. 추론 시간, peak RSS, 배치 불변성
7. 생성 산출물의 SHA-256과 재현 명령

실패한 실험도 결론과 폐기 이유를 `reports/`에 남겨 주세요. 실패 기록은 같은 방향을 반복하지 않게 해 주는 중요한 팀 자산입니다.

## 커밋은 필요한 파일만 선택해 주세요

먼저 변경 내용을 확인해 주세요.

```powershell
git status
git diff
```

그 다음 필요한 파일만 명시해서 추가해 주세요.

```powershell
git add -- src/example.py tests/test_example.py reports/example.md
git commit -m "exp: evaluate core reliability router"
```

초보자 실수를 줄이기 위해 `git add .` 또는 `git add -A`는 사용하지 않는 것을 팀 규칙으로 권장합니다.

## 데이터와 인증정보를 보호해 주세요

- DACON 원본 데이터는 저장소나 GitHub Release에 올리지 마세요.
- 모든 협업자는 동일한 공식 DACON 팀에 등록되어 있어야 합니다.
- `.env`, API token, 개인 쿠키와 계정정보를 공유하거나 커밋하지 마세요.
- 인증이 필요한 제출은 지정된 제출 담당자의 로컬 환경에서만 수행해 주세요.
- 모델, OOF, 제출 ZIP은 Git 이력이 아닌 합의된 private artifact 저장소를 이용해 주세요.
- artifact를 전달할 때는 반드시 파일명, 크기와 SHA-256을 함께 기록해 주세요.

커밋 전에는 다음 감사를 실행해 주세요.

```powershell
python scripts/audit_repository.py --include-untracked
```

## 제출 후보 승격 조건

다음 조건을 모두 통과한 후보만 DACON 제출 대상으로 검토해 주세요.

- 사전에 정한 엄격한 시계열 전이에서 일관되게 개선되어야 합니다.
- 도메인과 월별 결과에 치명적인 회귀가 없어야 합니다.
- bootstrap 개선 확률과 하위 5%가 충분히 양수여야 합니다.
- 테스트 행 순서와 배치 크기에 따라 결과가 달라지지 않아야 합니다.
- 공식 ZIP 구조와 시간·메모리·offline 추론 제한을 통과해야 합니다.
- 부모 ZIP과 변경 파일의 정확한 계보를 확인해야 합니다.

제출 후에는 결과를 즉시 `reports/submissions.csv`에 추가하고, 최고점이 갱신되면 `README.md`와 `docs/PROJECT_STATUS.md`도 함께 갱신해 주세요.
