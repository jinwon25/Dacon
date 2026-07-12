# 대회 적합성·오픈소스 검토

검토일: 2026-07-12

## 1. 현재 적합성 판단

| 요구사항 | 현재 상태 | 판단 | 조치 |
|---|---|---|---|
| 실제 월드컵 데이터 활용 | StatsBomb 경기 ID 3857262·3869321의 실제 이벤트와 FIFA 2026 경기 54 공식 사후 보고서 사용 | 적합 | 이벤트 좌표·전체 경기 집계·모델값을 분리 표시 |
| 직접 조작하는 감독 경험 | 선수·상대 드래그, 포메이션, 개인 역할, 팀 지시, 교체, A/B 비교 구현 | 적합 | 코칭 워크플로 용어로 전문성 강화 |
| 동적 인터랙션 | 모든 선택에 따라 비교 지표와 리포트가 갱신 | 적합 | 실제 결과 예측으로 오해하지 않게 `시나리오 비교`로 명명 |
| 웹에서 무설치 실행 | Vite 정적 웹앱 | 적합 | 배포 URL 회귀 테스트 필요 |
| 심사자 API 키 불필요 | 실행 시 외부 API 호출 없음 | 적합 | 검증한 요약 데이터를 정적 번들로 제공 |
| 저작권·라이선스 | 자체 CSS/SVG, 선수 사진·엠블럼 미사용 | 적합 | 데이터 출처와 StatsBomb 로고 필수 표시 |

범용 전술 스튜디오는 사용자가 경기 맥락과 선수를 직접 입력하는 수동 구조 점검 도구이며 실제 데이터 분석으로 표시하지 않습니다. 대회에서 요구하는 실제 월드컵 데이터 경험은 `실제 경기 분석` 모드가 담당합니다. 두 모드가 같은 서비스 안에서 연결되므로, 사용자는 실제 데이터 시나리오로 조작법을 익힌 뒤 자신의 경기 전술안을 자유롭게 만들 수 있습니다.

## 2. 가장 큰 위험과 수정 원칙

### 실제 데이터와 모델링 데이터의 혼동

기존 프로토타입은 실제 경기 맥락을 사용하지만 선수 능력치와 경기 결과가 휴리스틱이어서 `실제 데이터 기반 예측`으로 오해될 수 있었습니다. 다음 세 층을 명확히 구분합니다.

1. **관측값:** StatsBomb 이벤트에서 집계한 패스, 진입, 슈팅, 압박
2. **사용자 입력:** 포메이션, 위치, 역할, 팀 지시, 교체
3. **시나리오 점수:** 관측값을 기준선으로 사용자 입력에 가중치를 적용한 비교값

시나리오 점수는 승률·득점 확률·실제 결과 예측으로 표현하지 않습니다.

StatsBomb의 일반 이벤트 좌표는 22명의 연속 위치 추적 데이터가 아닙니다. 볼 흐름 재생은 패스·운반·슈팅의 시작점과 끝점을 시간순으로 보간한 표현이며 화면에 `연속 트래킹이 아님`을 명시합니다. StatsBomb 360을 쓰는 화면은 이벤트 순간 카메라에 보이는 선수의 프리즈프레임으로만 설명합니다.

2026 남아프리카공화국–대한민국전은 FIFA Training Centre가 공개한 `Post Match Summary Report`의 전체 경기 집계와 패스 매트릭스를 사용합니다. 64분 시점의 원시 이벤트 좌표나 연속 트래킹 데이터는 공개 자료에서 확보하지 못했으므로 생성하지 않습니다. 선수 위치는 포메이션 기반 참조 배치이며, 64분 개입안은 공식 보고서를 사후 검토하는 RE:TACTIC의 전술 재구성입니다.

### 게임형 결과의 과장

임의의 최종 스코어보다 `전술안 채택/조건부 채택`, 이점, 리스크, 전제조건을 제시하는 코칭 의사결정 메모가 실제 사용자에게 유용합니다. 기존 감독 성향 카드는 제거하고 인쇄 가능한 전술 개입 메모로 교체했습니다.

## 3. 사용 권장 데이터·도구

### 우선 사용

- [StatsBomb Open Data](https://github.com/statsbomb/open-data)
  - 2022 FIFA 월드컵 이벤트·라인업 데이터를 JSON으로 제공
  - 공개·공유하는 분석에는 StatsBomb 출처 명시와 로고 사용 요구
  - 현재 서비스의 대표 경기 데이터로 채택
  - 해커톤 공개 프로토타입 이후 유료·상용 코칭 서비스로 전환할 경우 StatsBomb에 별도 이용 범위를 확인
- [FIFA Training Centre · South Africa–Korea Republic Post Match Summary](https://www.fifatrainingcentre.com/media/native/tournaments/fifa-world-cup/2026/PMSR-M54-RSA-V-KOR.pdf)
  - 2026년 6월 24일 경기의 공식 전체 경기 통계, 국면 비중, 패스 매트릭스 사용
  - 원본 PDF를 저장소에 복제하지 않고 수치와 출처 링크만 정적 데이터로 구성
  - FIFA 로고나 보고서 그래픽을 복제하지 않고 텍스트 출처 표기만 사용
- [SkillCorner Open Data](https://github.com/SkillCorner/opendata)
  - MIT 라이선스의 방송 영상 기반 트래킹 샘플과 튜토리얼 제공
  - 실제 볼·선수 연속 이동, 오프더볼 런, 피치 컨트롤의 연구 데이터로 가장 명확한 후보
  - 월드컵 데이터가 아니므로 현재 대표 시나리오에는 섞지 않고 범용 스튜디오의 익명 학습 예제에만 향후 사용
- [kloppy](https://github.com/PySport/kloppy)
  - StatsBomb, Metrica 등 이벤트·트래킹 포맷과 좌표계를 표준화
  - BSD-3-Clause 라이선스
  - 여러 데이터 공급자를 함께 쓸 때만 도입 권장
- [socceraction](https://github.com/ML-KULeuven/socceraction)
  - SPADL 변환, xT·VAEP 기반 행동 가치 계산
  - MIT 라이선스
  - 2차 버전에서 전술 근거를 고도화할 때 적합
- [mplsoccer](https://github.com/andrewRowlinson/mplsoccer)
  - StatsBomb 데이터 로딩과 피치·히트맵·패스맵 생성
  - MIT 라이선스
  - 기획서용 정적 분석 이미지 생성에 유용
- [floodlight](https://github.com/floodlight-sports/floodlight)
  - 이벤트·트래킹 데이터 처리, 공간 통제·대사 파워 등 분석
  - MIT 라이선스
  - 실제 추적 데이터가 확보될 때 검토
- [LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)
  - 피치 컨트롤, EPV, 패스 옵션 평가의 교육용 구현
  - 코드 MIT 라이선스
  - 알고리즘 개념 참고용이며 그대로 복제하지 않고 출처 명시

### 주의해서 사용

- [Metrica Sports Sample Data](https://github.com/metrica-sports/sample-data)
  - 동기화된 익명 이벤트·트래킹 샘플 제공
  - 공개 사용 시 출처 표기를 요청하지만 저장소에 명시적인 OSI 라이선스가 보이지 않음
  - 현재 월드컵 시나리오와 섞지 않고, 피치 컨트롤 기능 연구·프로토타이핑에만 사용
- [football-data.org](https://www.football-data.org/pricing)
  - 무료 플랜은 일정·결과·순위 중심이며 점유율·슈팅 같은 상세 통계는 유료 부가 기능
  - API 키와 런타임 호출이 필요하므로 이번 제출물에는 사용하지 않음
- FIFA+, 방송사 영상, 선수 사진, 국가대표 엠블럼
  - 참고·검증에는 사용할 수 있으나 웹서비스에 복제하지 않음

대회 규칙상 실제 선수 이름·포지션·국가를 참고해 직접 JSON을 구성하는 것은 허용됩니다. 따라서 선수 명단·역할·전술 템플릿은 서비스용 로컬 TypeScript/JSON 데이터로 직접 구성하고, 실제 관측 통계와 명확히 분리합니다. 본 프로젝트는 특정 게임의 능력치, 선수 사진, FIFA 로고, 국가대표 엠블럼을 사용하지 않습니다. `2022` 경기를 선택한 이유는 연도 제한이 없고 출처·재현 방법이 명확한 실제 이벤트 데이터가 공개되어 있기 때문입니다.

> 이 문서는 대회 제출을 위한 실무 검토이며 법률 자문이 아닙니다. 향후 유료 서비스나 구단 계약 단계에서는 데이터·상표 이용 조건을 권리자에게 다시 확인해야 합니다.

## 4. 채택하지 않는 접근

- 키가 필요한 상용 API를 브라우저에서 직접 호출
- 라이선스가 불명확한 게임 선수 능력치 복제
- Football Manager, TacticalPad, Coach Paint의 UI·아이콘·그래픽 모방
- 단일 경기 데이터로 학습했다고 주장하는 AI 승률 모델
- 근거 없이 생성형 AI가 전술을 확정적으로 추천하는 기능

## 5. 출처 표시 문구

> Match event data: StatsBomb Open Data. Derived metrics were calculated by RE:TACTIC from match 3857262. Scenario scores are heuristic comparisons, not predictions of real match outcomes.

서비스 하단과 README에 위 문구와 StatsBomb 로고를 표시합니다.

## 6. 데이터 재현

다음 명령은 StatsBomb 기반 두 시나리오의 팀별 관측값과 인플레이 점유 추정을 원본 이벤트 JSON에서 다시 계산합니다. 2026 시나리오는 FIFA 공식 PDF의 공개 집계값을 화면과 코드에서 교차 확인합니다.

```bash
node scripts/extract-statsbomb-evidence.mjs
```

원본은 저장소에 복제하지 않고, 웹서비스에는 검증한 소규모 파생 요약값만 포함합니다.

## 7. 실제 포함된 오픈소스 패키지

| 패키지 | 용도 | 라이선스 |
|---|---|---|
| React / React DOM 18.3.1 | 사용자 인터페이스 | MIT |
| Vite 6.4.3 | 개발·프로덕션 빌드 | MIT |
| TypeScript 5.7.3 | 정적 타입 검사 | Apache-2.0 |
| html-to-image 1.11.13 | 전술안 DOM을 PNG로 변환 | MIT |
| country-flag-icons 1.6.20 | 국가 표기를 위한 일관된 SVG 국기 | MIT |

`html-to-image`는 클라이언트에서 사용자가 만든 전술안만 이미지로 변환하며 외부 서버로 내용을 전송하지 않습니다. 의존 패키지의 원문 라이선스는 설치된 패키지와 각 공식 저장소에서 확인할 수 있습니다.
