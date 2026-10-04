# 공개 전 검토 기록

검토 기준일: 2026-10-04.

## 투구 프로젝트 공개 판단

대회·코드 검증 기간은 종료됐습니다. 개별 규칙·동의사항·FAQ에서 종료 후 자체 코드 공개를 금지하는 별도 조항을 확인하지 못했습니다. 다른 대회에서 DACON.GM이 제공한 종료 후 GitHub 공개 권장 안내를 함께 검토해, **데이터와 비공개 산출물을 제외한 코드·설명 공개가 가능한 것으로 판단**했습니다. 주최 측이 이 저장소를 개별 승인했다는 의미는 아닙니다. 근거는 이 문서 아래의 공식 자료에 정리했습니다.

원본 데이터·데이터 설명 문서·공식 baseline, 선수·경기 연결표, 행별 예측/OOF, 학습 모델(JSON 포함), 제출 ZIP, 비공개 링크와 노트북 실행 출력을 과거 커밋에서도 제거했습니다. 데이터 파생 자산까지 제외한 것은 데이터 재배포 위험을 줄이기 위한 이 저장소의 보수적인 공개 정책입니다. 규정이 모든 모델의 공개를 일률적으로 금지한다고 단정하지 않습니다.

## 발견한 문제와 조치

| 발견 | 조치 |
|---|---|
| 현재 파일 검사로는 과거 이력·JSON 모델·연결표를 놓침 | 전체 이력 검사와 파일 내용 검사 추가, 새 원격 저장소에 정리한 main만 게시 |
| 투구 CI가 프로젝트 하위 `.github/`에 위치하고 이전 테스트 경로 참조 | 루트 Actions로 이동, 현재 테스트 전체 실행 |
| v148·1161을 “현재 champion”으로 표시한 과거 문서 | 최종 v345를 명시하고 과거 문서 작성 시점 구분 |
| 공개하지 않은 입력을 clone에 포함된 것처럼 안내 | 데이터 없는 검사·체크포인트 재조립·신규 학습 검증을 구분 |
| 창고 최종 SLSQP 설명과 자동 최소 OOF 선택 코드 불일치 | 최종 레시피에서 `--method global_slsqp` 명시 |
| 풍력 README의 긴 과거 영문 기록 | 상세 연구 이력으로 분리 |
| 전술 프로젝트의 완료/출품 표현과 체크리스트 불일치 | 구현된 프로토타입과 미확인 최종 제출 상태 구분 |
| 삭제된 모기 코드 공유 글의 포팅 소스에 재배포 라이선스 근거 없음 | 포팅 소스 `v148_cree_xy2.py`를 전체 공개 이력에서 제외하고 재현 제약 명시 |
| 프론트엔드 빌드 의존성 4건 취약점 | 기존 허용 버전 범위 내 lockfile 갱신, npm audit 0건 |

## 검증 범위

루트 공개 정책 검사는 현재 파일과 모든 로컬 Git ref의 파일·커밋 메시지를 검사합니다. CSV는 집계 보고 경로만 허용하고 식별자·정답 열을 차단하며 학습 JSON과 노트북 출력을 검사합니다. 자동 검사는 패턴 기반이므로 모든 저작권·데이터 누출이나 모델링 규정 위반을 증명하지 않습니다.

로컬 검증 결과: 공개 정책 회귀 테스트 6개 통과, 투구 최초 로컬 테스트 735개 통과·25개 자산 의존 skip (공개 정책 회귀 테스트 추가 후 원격 CI는 736개 통과·25개 skip), 창고 블렌드 회귀 테스트 1개 통과, RE:TACTIC 타입·배포 빌드 통과 및 E2E 13개 통과·1개 중복 뷰포트 skip. 원본 데이터와 체크포인트가 필요한 전체 학습·숨겨진 평가 점수는 이번 공개 검토에서 재실행하지 않습니다. 기존 재현 보고는 당시 조건의 증거로 유지합니다. GitHub Actions에서 동일한 검사를 반복할 수 있습니다.

## 규정·구성 참고 자료

확인일: 2026-10-04. 문구를 그대로 복제하지 않고 프로젝트에 맞게 요약·적용했습니다.

### 공식 공개 근거

- [투구 대회 일정](https://dacon.io/competitions/official/236743/overview/schedule): 온라인 대회 9월 2일 종료, 코드 검증 9월 11일 종료. Phase 3는 별도 대회입니다.
- [투구 대회 규칙](https://dacon.io/competitions/official/236743/overview/rules): 공식 데이터만 사용, 외부 API 제한, 평가 행 간 독립 추론.
- [투구 동의사항](https://dacon.io/competitions/official/236743/overview/agreement): 데이터 목적 외 이용·비참가자 배포 제한 및 코드 공유·제3자 권리 조건.
- [DACON.GM의 GitHub 공개 안내](https://dacon.io/competitions/official/236027/talkboard/407199): 특별한 주최 측 지침이 없으면 대회 종료 후 GitHub 공개를 권장한 운영진 답변. **다른 대회에서의 일반 안내이며 이 대회에 대한 개별 승인서는 아닙니다.**
- [투구 공식 FAQ](https://dacon.io/competitions/official/236743/talkboard/417082), [평가](https://dacon.io/competitions/official/236743/overview/evaluation), [리더보드](https://dacon.io/competitions/official/236743/leaderboard).
- [모기 대회 규칙](https://dacon.io/competitions/official/236716/overview/rules), [풍력 대회 규칙](https://dacon.io/competitions/official/236727/overview/rules).

### 저장소 구성 참고

- [GitHub README 안내](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes): 목적·시작 방법·참여자 정보를 루트에 두고 상세 문서는 연결.
- [GitHub 민감 정보 제거 안내](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository): 현재 파일 삭제만으로 과거 이력의 내용이 제거되지 않음.
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/): 데이터·소스·실험 결과·모델을 분리하는 구조 참고.
- [DACON 공식 대회 예제](https://github.com/Dacon-official/competitions): 코드와 공식 데이터 취득 경로를 구분하는 방식 참고.
- [Data Analyst Portfolio 사례](https://github.com/sebastianmukuria/DataAnalystPortfolio), [Data Analytics Portfolio 사례](https://github.com/zamimammadov/DATA-ANALYTICS-PORTFOLIO): 프로젝트 목록과 분석 질문·방법·결과 중심의 읽기 경로 참고. 해당 코드나 결과를 차용하지 않았습니다.

### 출처와 권리

프로젝트 내 기존 출처·라이선스 표시는 유지합니다. RE:TACTIC의 라이선스와 도구별 라이선스는 해당 경로에 적용됩니다. 저장소 전체에 하나의 새 라이선스를 덮어씌우지 않습니다. 모기 회전물리 모델의 기존 공개 코드 참고 및 원 게시물 삭제 사실은 해당 README에 명시돼 있습니다. 재배포 허가를 확인할 수 없는 모기 `v148_cree_xy2.py` 포팅 소스는 공개 이력에서 제외했습니다. 규정 검토는 개별 원저작물의 라이선스 허가를 대신하지 않습니다.

### 본인 계정의 다른 공개 저장소와 비교

이커머스 고객 세분화, KBO 투수 피로도, 취업 준비 텍스트 마이닝, KBO Predictor의 README와 루트 구성을 읽기 전용으로 참고했습니다. 해당 레포의 질문·결과·역할·해석 한계 중심 설명을 Dacon에도 맞췄습니다. 다른 레포를 수정하거나 그 결과를 대회 성과로 합산하지 않았습니다.

원본 분석용 `scripts/`·`tests/`는 이 레포들에서도 검증 근거로 사용합니다. Dacon 루트에 추가한 공개 검사만 `.github/`로 묶었으며 각 프로젝트의 고유 테스트·문서는 유지합니다.
