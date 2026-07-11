# 상용·유사 서비스 벤치마크

조사일: 2026-07-11

## 1. 벤치마크 요약

| 제품·사례 | 강점 | RE:TACTIC에 반영 | 그대로 따라 하지 않을 것 |
|---|---|---|---|
| [TacticBoard](https://www.tacticboard.app/about) | 즉시 시작하는 드래그 앤 드롭, 포메이션 분석, 다중 프레임 전술 | 설명 없이 옮길 수 있는 선수 토큰, 빠른 포메이션 프리셋, 전술 위험 즉시 피드백 | 자유 드로잉과 세션 저장 등 코치용 범용 기능 |
| [TacticalPad](https://www.tacticalpad.com/new/index.php) | 라인업·드릴·전술 애니메이션을 한 도구에 통합 | 전술 변화가 실제 움직임으로 이어지는 짧은 애니메이션 | 전문가용 편집 도구 수준의 복잡한 타임라인 |
| [easy2coach Tactics](https://www.easy2coach.net/en/tactics/) | 포메이션·세트피스 템플릿, 화살표, 다중 애니메이션 단계, 공유 | 4개 빠른 시작 템플릿, 패스·침투 도구, 최대 5개 장면 재생 | 수백 개 아이콘과 콘텐츠 관리 기능 |
| [Coach Paint](https://www.coachpaint.com/) | 선수 추적, 포메이션 도구, 3D 그래픽 등 방송형 텔레스트레이션 | 패스 길, 위험 공간, 수적 우위를 한눈에 보이는 오버레이로 표현 | 영상 업로드·자동 추적·3D 등 개인 개발 범위를 넘는 기능 |
| [Football Manager](https://www.footballmanager.com/features/truer-football-motion-match-authenticity-positional-play) | 포메이션, 선수 역할, 팀 지시가 매치 엔진의 움직임과 연결 | 포지션과 역할을 분리하고 전술 선택이 결과 지표에 영향을 주게 설계 | 방대한 선수 DB와 전체 경기 매치 엔진 |
| [Football Manager 26 Visualiser](https://www.footballmanager.com/fm26/features/possession-out-possession-fm26s-new-tactical-evolution) | 공의 위치와 국면에 관련된 지시만 보여주는 시각적 전술 구조 | 중앙 피치 중심 구조와 선택 대상에 따라 바뀌는 우측 패널 | 항상 노출되는 다층 메뉴와 높은 정보 밀도 |
| [Sofascore](https://www.sofascore.com/) | 라인업·평점·히트맵·이벤트를 탭으로 단계화 | 선수·분석·전술안 탭과 짧은 비교 지표 | 경기장 주위에 모든 통계를 동시에 노출 |
| [TacticAI](https://deepmind.google/blog/tacticai-ai-assistant-for-football-tactics/) | 실제 축구 데이터를 바탕으로 전술 결과 예측과 대안 제안 | AI처럼 보이는 만능 채팅보다 근거가 보이는 `코치 제안`과 대안 비교 | 학습 모델을 썼다고 과장하거나 검증 불가능한 승률 제시 |

## 2. UX 원칙

### 첫 조작까지 10초

TacticBoard의 장점은 별도 설치나 긴 학습 없이 바로 선수를 옮길 수 있다는 점입니다. RE:TACTIC도 긴 서비스 소개 대신 경기 스코어와 목표를 보여주고 곧바로 `지휘 시작` 버튼을 제공합니다.

### 설정과 결과의 거리를 짧게

Football Manager처럼 많은 설정을 제공하되, 핵심 MVP에서는 압박·폭·템포·위험 감수 네 가지로 한정합니다. 조절 즉시 공격 위협도, 중원 장악력, 역습 노출도, 체력 부담이 변해 설정의 의미를 이해할 수 있게 합니다.

### 데이터는 해석과 함께

Coach Paint의 방송형 시각화처럼 차트를 읽는 법을 배우지 않아도 위험 공간과 패스 길을 알아볼 수 있어야 합니다. 모든 데이터 카드에는 `그래서 무엇을 해야 하는가`를 한 문장으로 붙입니다.

### 시뮬레이션보다 선택의 설득력

개인 프로젝트에서 실제 축구 전체를 재현하는 매치 엔진은 범위와 신뢰성 모두 위험합니다. 대신 실제 경기의 특정 시점 이후를 3~5개 전술 분기로 구성하고, 각 결과에 영향을 준 선택을 리포트에서 설명합니다.

## 3. 차별화 문장

> 기존 전술 보드가 생각을 그리는 도구라면, RE:TACTIC은 실제 경기의 한복판에서 그 생각의 결과까지 체험하는 서비스다.

## 4. 데이터 후보

- [StatsBomb Open Data](https://github.com/statsbomb/open-data): 2022 월드컵 경기 이벤트·라인업 데이터. 공개·공유 시 StatsBomb 출처와 로고 표시 필요.
- [StatsBomb 2022 World Cup 공개 안내](https://statsbomb.com/news/statsbomb-release-free-2022-world-cup-data/): 2022 월드컵 전체 데이터 공개 범위 확인.
- [FIFA 2022 공식 스쿼드 목록](https://fdp.fifa.org/assetspublic/ce44/pdf/SquadLists-English.pdf): 선수명, 포지션, 당시 소속팀 등 명단 교차 확인.
- [FIFA+ 한국 대 포르투갈 전체 경기](https://www.plus.fifa.com/en/content/68827163-5c2f-4f2c-9553-97ce7159ecaf): 시나리오 타임라인과 실제 장면 확인용. 영상·이미지는 서비스에 복제하지 않음.

## 5. 저작권·표현 가이드

- 방송 영상, 선수 사진, FIFA·국가대표 엠블럼을 무단 사용하지 않습니다.
- 국기는 이모지 또는 직접 제작한 단순 색상 요소로 표현합니다.
- 피치, 선수 토큰, 히트맵은 CSS/SVG로 직접 제작합니다.
- 실제 데이터와 서비스용 추정 지표를 시각적으로 구분합니다.
- 데이터 출처와 가공 방식은 서비스 내 `데이터 노트`와 README에 표시합니다.

## 6. 이번 MVP에서 채택하는 기능

1. 클릭 즉시 시작하는 단일 경기 챌린지
2. 드래그 가능한 선수와 네 가지 포메이션 프리셋
3. 포지션별 선수 역할 선택
4. 팀 전술 슬라이더와 실시간 지표 변화
5. 한 번의 교체 선택
6. 선택 근거를 설명하는 코치 피드백
7. 채택 판단, 이점, 리스크, 운영 조건을 담은 전술 개입 메모
8. 템플릿 → 배치 → 경로 → 장면 → 재생으로 이어지는 범용 코칭 흐름
9. 텍스트로 끝나지 않고 미리보기·적용까지 제공하는 규칙 기반 코치 제안
