# 팀원 온보딩 체크리스트

팀 organization 저장소 `Lg-Aimers-chungang/hackathon`을 기준으로 아래 순서대로 진행한다.

## 저장소 관리자

- [ ] 두 팀원이 같은 공식 DACON 팀에 등록되어 있는지 확인합니다.
- [ ] GitHub organization과 private 저장소에 팀원을 초대합니다.
- [ ] 초대 권한은 우선 `Write`로 설정합니다.
- [ ] 두 팀원이 초대를 수락했는지 확인합니다.
- [ ] 각 팀원의 첫 clone과 CI 테스트 성공을 확인합니다.
- [ ] 같은 공식 DACON 팀으로 병합된 것을 확인한 뒤 `Google Drive 팀 폴더`에 팀원 계정을 개별 초대합니다.
- [ ] Drive의 `일반 액세스`가 `제한됨`인지 확인하고 챔피언 ZIP과 최소 모델 artifact를 인계합니다.
- [ ] 팀원 한 명이 파일을 내려받아 `docs/ARTIFACT_HANDOFF.md`의 SHA-256과 일치하는지 확인합니다.
- [ ] DACON 제출 담당자 한 명과 백업 담당자 한 명을 정합니다.

## 새 팀원

- [ ] GitHub 계정의 2단계 인증을 켭니다.
- [ ] 저장소 초대를 수락합니다.
- [ ] 저장소를 clone하고 Python 환경을 설치합니다.
- [ ] DACON에서 데이터를 직접 내려받습니다.
- [ ] `data/README.md`의 SHA-256과 비교합니다.
- [ ] `python scripts/audit_repository.py --include-untracked`를 통과합니다.
- [ ] 데이터 비의존 테스트를 통과합니다.
- [ ] 연습 branch를 만들고 작은 문서 수정 PR을 한 번 진행합니다.

## 첫 연습 Pull Request

처음부터 모델 코드를 수정하기보다 각자 `docs/onboarding-<username>.md` 같은 짧은 문서를 branch에서 추가해 보는 것을 권장합니다. 이 과정에서 branch, commit, push, PR, review와 merge 흐름을 안전하게 연습하실 수 있습니다.

## 현재 저장소 제약

- `main`에 직접 push하지 않고 개인 feature branch와 Pull Request를 사용한다.
- 필요한 파일만 `git add -- <경로>`로 추가하고, 데이터·모델·ZIP·인증정보가 포함되지 않았는지 로컬 저장소 감사를 실행한다.
- branch protection과 Secret Scanning 설정 여부와 무관하게 위 규칙을 팀 기본 절차로 유지한다.
