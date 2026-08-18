# 산출물 정책

이 디렉터리는 학습 모델, OOF 예측과 실험 캐시를 보관하며 Git에서는 제외한다. 최종 제출 ZIP은 여기에 두지 않는다.

## 현재 기준

- champion: 프로젝트 루트의 `submit_v26.zip` (공식 DACON 제출 이력 대조로 2026-08-18 정정. `README.md`와 `reports/target1170_followup_20260817.md`의 v27 champion 표기는 오류이며, 근거는 [`../reports/target1170_followup_20260817.md`](../reports/target1170_followup_20260817.md) 상단 정정 안내와 [`../notebooks/v26_champion_reproduction.ipynb`](../notebooks/v26_champion_reproduction.ipynb)를 참고한다.)
- champion SHA-256: `8BE26E156A91C1FA5E9989EEE5043DB5992A960DF35A53D8A9BFEF7E6825D096`
- v26 직접 부모: `../submit_v25.zip` (SHA-256 `60CF13B28AF01B24A49E2B03E13161F2E04A1C09647D8DA2CFD95407ED9335B1`)
- v26 검증 기준선(조부모): `../submit_v22.zip` (SHA-256 `FB12FC236B82E6D72D1A870F3F8C52CBD8C5D79D31E3F0E98193E0F648BB47B7`)
- 재빌드·검증 진입점: [`../notebooks/v26_champion_reproduction.ipynb`](../notebooks/v26_champion_reproduction.ipynb)

`v25_postbreak_anchor_20260817_01/`과 `v25_postbreak_anchor_audit_20260817_01/`은 v25/v26 재패키징과 최종 학습 해시 검증에 필요하므로 유지한다. `state_mode_joint_final_20260816/` 등 v14→v19 계보 artifact도 과거 재현용으로 보존한다. 이름에 `tmp_`가 붙은 압축 해제본, 빈 실행 폴더, 캐시와 빈 로그는 정리 대상이다.

나머지 날짜별 폴더는 과거 실험 재현 자료다. 공간 확보를 위해 삭제할 때는 먼저 연결된 `reports/` 문서와 manifest가 있는지 확인한다.

## 보관 원칙

- 챔피언 제출 ZIP과 필수 모델: 팀이 합의한 private artifact 저장소
- 중간 OOF와 대규모 캐시: 각 실험 담당자 로컬 또는 DVC/object storage
- Git에 남길 내용: 생성 명령, 입력 기준선, SHA-256, 크기, 검증 결과
