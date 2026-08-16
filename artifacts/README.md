# 산출물 정책

이 디렉터리는 학습 모델, OOF 예측과 실험 캐시를 보관하며 Git에서는 제외한다. 최종 제출 ZIP은 여기에 두지 않는다.

## 현재 기준

- champion: 프로젝트 루트의 `submit_v19.zip`
- champion SHA-256: `B12070D8017AE6A78BFC056F392BACDA931F8ED7439AA9821880FC986CE962B2`
- v19 최종 모델: `state_mode_joint_final_20260816/`
- v19 로컬 후보 요약: `state_mode_joint_20260816_02/summary.json`
- v19 직접 부모: `../submissions/history/submit_v17.zip`

`state_mode_joint_final_20260816/`은 v19 재패키징과 최종 학습 해시 검증에 필요하므로 유지한다. 이름에 `tmp_`가 붙은 압축 해제본, 빈 실행 폴더, 캐시와 빈 로그는 정리 대상이다.

나머지 날짜별 폴더는 과거 실험 재현 자료다. 공간 확보를 위해 삭제할 때는 먼저 연결된 `reports/` 문서와 manifest가 있는지 확인한다.

## 보관 원칙

- 챔피언 제출 ZIP과 필수 모델: 팀이 합의한 private artifact 저장소
- 중간 OOF와 대규모 캐시: 각 실험 담당자 로컬 또는 DVC/object storage
- Git에 남길 내용: 생성 명령, 입력 기준선, SHA-256, 크기, 검증 결과
