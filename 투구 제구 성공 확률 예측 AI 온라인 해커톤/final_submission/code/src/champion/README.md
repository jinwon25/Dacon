> Historical lineage note: the final selected version is **v345 (1182.9497)**. The v148 instructions below describe the earlier stage and require excluded private artifacts.

# 배포 계보 모듈

이 디렉터리는 최종 v345 계보에서 사용된 초기·중간 세대의 패키지 조립 모듈을
보존한다. 파일명과 버전 번호는 재현 명령의 호환성을 위해 유지한다.

- `src/champion/`: 실제 배포 계보에서 사용된 빌더와 런타임 모듈
- `src/archive/`: 비교 실험 및 이전 세대 구현
- `inference/script.py`: 최종 v345 추론 동작의 기준

초기 세대의 개별 명령을 현재 최종 모델 실행 명령으로 해석하지 않는다. 최종 재현은
프로젝트 루트의 `03_REPRODUCE.md`를 따르고, 배포 단계와 선택 근거는
`code/DEVELOPMENT_RECORD.md`, 구성요소별 검증 상태는 `verification/`에서 확인한다.

빌더는 입력 ZIP·설정·출력 디렉터리를 명시적으로 받는다. 기존 체크포인트를 덮어쓰지
말고 새 출력 디렉터리를 사용한다. 최종 추론 코드와 가중치는 재현 검증 과정에서
변경하지 않는다.
