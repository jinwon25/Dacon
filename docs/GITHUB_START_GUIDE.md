# GitHub 처음 시작하기

이 문서는 GitHub와 Git이 익숙하지 않은 팀원을 위한 안내입니다. 명령을 외우실 필요는 없습니다. 작업을 시작할 때와 마칠 때 이 문서의 순서를 그대로 따라 주세요.

## 자주 나오는 용어

| 용어 | 뜻 |
|---|---|
| 저장소(repository) | 프로젝트 파일과 변경 이력을 보관하는 공간입니다. |
| clone | GitHub의 저장소를 내 컴퓨터로 처음 내려받는 작업입니다. |
| branch | 다른 팀원의 안정적인 코드에 영향을 주지 않고 작업하는 별도 작업선입니다. |
| commit | 변경 내용을 설명과 함께 하나의 저장 지점으로 남기는 작업입니다. |
| push | 내 컴퓨터의 commit을 GitHub로 올리는 작업입니다. |
| Pull Request(PR) | 내 branch의 변경을 `main`에 합쳐 달라고 요청하는 검토 화면입니다. |
| merge | 검토가 끝난 PR을 `main`에 합치는 작업입니다. |
| CI | push된 코드의 기본 테스트를 GitHub가 자동으로 실행하는 기능입니다. |

## 처음 한 번만 준비해 주세요

1. GitHub 계정을 만든 뒤 username을 저장소 관리자에게 알려 주세요.
2. 가능하면 GitHub 2단계 인증을 켜 주세요.
3. 저장소 초대 메일을 받고 초대를 수락해 주세요.
4. Git과 Python 3.11을 설치해 주세요.
5. 원하는 작업 폴더에서 저장소를 clone해 주세요.

```powershell
git clone https://github.com/Lg-Aimers-chungang/hackathon.git
cd pitch-control-probability
git config user.name "GitHub 표시 이름"
git config user.email "GitHub에 등록한 이메일 또는 noreply 이메일"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

GitHub 로그인 때문에 clone이 실패하면 GitHub Desktop을 사용하셔도 됩니다. GitHub 비밀번호를 터미널에 직접 입력하지 마시고, 브라우저 로그인이나 GitHub Desktop의 인증 화면을 이용해 주세요. 중요한 점은 저장소를 내려받은 뒤 프로젝트 폴더 안에서 명령을 실행하는 것입니다.

위의 `user.email`에는 공개를 원하지 않는 개인 이메일 대신 GitHub의 noreply 이메일을 사용하셔도 됩니다. GitHub의 `Settings > Emails`에서 확인하실 수 있습니다.

## 데이터를 준비해 주세요

원본 데이터는 GitHub에 없습니다. DACON에서 직접 내려받은 다음 `data/`에 넣어 주세요.

```text
data/train.csv
data/trackman_history.csv
data/test.csv
data/sample_submission.csv
```

파일이 같은지 확인하려면 [`../data/README.md`](../data/README.md)의 SHA-256과 비교해 주세요.

## 작업을 시작할 때마다 실행해 주세요

```powershell
git switch main
git pull --ff-only
git status
```

`git status`에 변경 파일이 없어야 안전하게 새 작업을 시작할 수 있습니다. 그 다음 본인의 branch를 만들어 주세요.

```powershell
git switch -c exp/my-first-experiment
```

이미 만든 branch로 돌아가려면 다음처럼 실행해 주세요.

```powershell
git switch exp/my-first-experiment
```

## 작업 중에는 자주 상태를 확인해 주세요

```powershell
git status
git diff
```

`git status`는 어떤 파일이 바뀌었는지 보여 줍니다. `git diff`는 파일 안에서 무엇이 바뀌었는지 보여 줍니다.

## 작업을 GitHub에 올려 주세요

먼저 테스트와 안전 감사를 실행해 주세요.

```powershell
python scripts/audit_repository.py --include-untracked
python -m pytest -q tests/test_metrics.py tests/test_features.py tests/test_repository_audit.py
```

필요한 파일만 선택해서 commit해 주세요.

```powershell
git add -- src/변경한파일.py tests/test_변경한파일.py research/reports/실험기록.md
git commit -m "exp: describe the experiment briefly"
git push -u origin exp/my-first-experiment
```

`git add .`와 `git add -A`는 원본 데이터나 비밀 파일을 실수로 포함할 수 있으므로 사용하지 마세요.

## Pull Request를 만들어 주세요

1. GitHub 저장소 페이지를 열어 주세요.
2. 방금 push한 branch 옆의 `Compare & pull request` 버튼을 눌러 주세요.
3. 자동으로 나타나는 양식에 가설, 검증 구간과 결과를 작성해 주세요.
4. 다른 팀원 한 명에게 리뷰를 요청해 주세요.
5. CI가 초록색으로 성공했는지 확인해 주세요.
6. 리뷰가 끝나면 squash merge해 주세요.

현재 private 저장소에서는 요금제 제한으로 branch protection을 강제할 수 없습니다. 실수로 `main`에 직접 push하지 않도록 꼭 branch와 PR을 이용해 주세요.

## 병합이 끝난 뒤 정리해 주세요

```powershell
git switch main
git pull --ff-only
git branch -d exp/my-first-experiment
```

GitHub 화면에서도 병합된 원격 branch를 삭제해 주세요.

## 문제가 생겼을 때 하지 말아야 할 일

원인을 모르는 상태에서는 다음 명령을 실행하지 마세요.

```text
git reset --hard
git clean -fd
git push --force
```

충돌이나 오류가 생기면 `git status` 출력과 실행했던 명령을 팀 채널에 공유해 주세요. 변경 파일을 삭제하거나 강제로 되돌리기 전에 다른 팀원과 먼저 확인해 주세요.
