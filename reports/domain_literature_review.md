# 한국 야구 제구 도메인 문헌 검토

외부 자료는 가설 설계에만 사용했으며 모델 학습·추론 데이터에는 결합하지 않았다.

## 1. KBO 2024 ABS 운영 규정

- 출처: [KBO ABS 및 피치클락 세부 운영 규정](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=9939), 2024.
- 검증된 사실: 2024 KBO 리그 ABS는 좌우로 홈 플레이트보다 각 2cm 넓고, 상하단은 타자 신장 비율과 홈플레이트 중간·끝면을 함께 사용한다. 포수 포구 방식과 무관하게 트래킹 좌표로 판정한다.
- 적용: `season`, `game_type`, count/handedness 상호작용과 2024 regime을 EDA했다.
- 기각: 실제 위치와 타자 신장이 메인 데이터에 없으므로 ABS zone 자체를 재구성하지 않았다.

## 2. KBO 2025 규정 변화

- 출처: [KBO 2025 규정·규칙 변화](https://www.koreabaseball.com/Kbo/League/GameManage2025.aspx), KBO.
- 검증된 사실: 2025 ABS 상·하단은 각각 신장 비율 0.6%p 하향됐고, KBO 리그 피치클락이 정식 적용됐다. 주자 유무에 따라 투구 시간이 25초/20초로 다르다.
- 적용: 주자·LI·이닝·카운트의 pressure interaction을 검토했다.
- 보류: 2025 rule indicator는 train에서 항상 0이므로 수동 효과값을 만들지 않았다. test 행이나 Public 점수로 효과를 역추정하지 않는다.

## 3. KBO 리그와 퓨처스리그 제도 차이

- 출처: [2024 제도별 KBO/퓨처스 도입 시기](https://www.koreabaseball.com/MediaNews/Notice/View.aspx?bdSe=9932), KBO.
- 검증된 사실: ABS는 퓨처스리그에서 2020년부터, KBO 리그에서는 2024년부터 운영됐다. 피치클락은 2024 퓨처스에서 먼저 적용됐다.
- 적용: game type을 하나의 전역 시즌 추세로 합치지 않고 익명 strata별 regime residual을 점검했다.
- 주의: 공식 대회 설명서는 `F`와 `R` 의미를 정의하지 않으므로 F=퓨처스로 확정하지 않았다.

## 4. KBO ABS 영향 분석

- 출처: [Analyzing the Impact of the Automatic Ball-Strike System in Professional Baseball](https://arxiv.org/abs/2407.15779), Scientific Reports 계열 연구, 2024/2025.
- 검증된 주장: 2024 ABS zone은 이전 인간 심판 zone보다 경계가 일관되고 직사각형에 가까웠으며, high/low 및 inside/outside 경계의 판정률과 handedness 차이가 바뀌었다. 저자들은 2-2 count에서 투구 위치·구종 변화도 분석했다.
- 적용: count×regime, handedness×regime을 한정 실험했다.
- 결과: explicit regime LGB는 2024에서 standalone blend를 악화해 최종 후보에서 제외했다.

## 5. 의도와 실행을 분리한 xCTRL

- 출처: [Separating Intent from Execution: xCTRL](https://arxiv.org/abs/2508.19184), 2025.
- 검증된 주장: 투수×구종×타자 손잡이별 위치 분포로 개인화된 의도를 추정하고, low-data context에는 상위 분포로 shrink한다.
- 적용: 개인별 과거 성공·가운데·reverse·ball 구성과 표본 신뢰도를 분리해서 보려는 원칙만 사용했다.
- 기각: 메인 데이터에 현재 구종·실제 위치·포수 target이 없고 TrackMan 선수 ID 조인이 0%이므로 xCTRL/GMM을 구현하지 않았다.

## 6. KBO 투구 유형·위치 예측

- 출처: [Prediction of pitch type and location using an ensemble of DNNs](https://journals.sagepub.com/doi/10.3233/JSA-200559), Journal of Sports Analytics, 2022.
- 검증된 주장: KBO 투구 의사결정은 승/무/패, count, 주자, 타자 등 상황에 따라 달라지고 투수·포수·벤치의 집단 결정이라는 점을 반영했다.
- 적용: count, score state, runner pressure, pitcher/batter handedness 상호작용을 사용했다.
- 한계: 포수 ID와 요구 코스가 없으므로 catcher effect를 억지로 복원하지 않았다.

## 7. Count와 제구 가능한 공의 선택

- 출처: [Using multi-class classification methods to predict baseball pitch types](https://journals.sagepub.com/doi/10.3233/JSA-170171), Journal of Sports Analytics, 2018.
- 검증된 주장: batter-ahead count에서는 볼넷을 피하기 위해 제어 가능한 공의 선택이 더 예측 가능했다.
- 적용: three-ball, two-strike, batter/pitcher-ahead를 명시적 범주로 만들었다.
- 결과: pressure feature 자체는 유의한 EDA span이 있었으나 기존 count categories를 크게 넘는 standalone 개선은 확인하지 못했다.

## 8. Pitch command biomechanics

- 출처: [Using Sensors for Player Development: Factors Related to Pitch Command](https://pmc.ncbi.nlm.nih.gov/articles/PMC9655623/), Sensors, 2022.
- 검증된 주장: 소규모 대학 투수 실험에서 forearm peak linear acceleration이 command와 연관됐고 velocity 관련 biomechanics와는 다른 변수가 중요했다.
- 적용: command와 stuff를 동일시하지 않고, TrackMan 물리량을 ID 연결 없이 league average로 대체하지 않았다.
- 기각: 센서·현재 투구 물리량은 제공되지 않는다.

## 9. 경기 내 피로와 투구 mechanics

- 출처: [The Impact of Fatigue on the Kinematics of Collegiate Baseball Pitchers](https://pmc.ncbi.nlm.nih.gov/articles/PMC4555605/), 2015.
- 검증된 주장: inning/game/season 진행과 pitch count가 일부 pitching kinematics 변화와 연관됐다.
- 적용: inning과 leverage는 사용했다.
- 기각: 메인 데이터에 game ID와 경기 내 투구 수가 없고 row_id 순서가 시간이라는 근거가 없어 fatigue count를 재구성하지 않았다.

## 10. 반복 투구와 accuracy

- 출처: [Alterations in pitching biomechanics and performance with increasing pitches](https://pubmed.ncbi.nlm.nih.gov/37574914/), PM&R, 2024.
- 검증된 주장: 반복 투구에 따른 피로가 kinetic chain과 proprioception을 방해해 velocity와 accuracy를 저하시킬 수 있다는 문헌을 종합했다.
- 적용: 최근 경기 성공률·가운데 비율의 변화와 변동성을 시험했다.
- 결과: domain-profile LGB는 네 fold blend에서 신호가 있었지만 최적 iteration 변동이 커 패키지 승격을 보류했다.

## 최종 채택

문헌상 가장 중요한 것은 “제구 능력은 개인 이력뿐 아니라 count·손잡이·경기 제도에 따라 관측되는 실행 정책이 달라진다”는 점이다. 현재 데이터에서 새로 확인된 가장 큰 신호는 game type별 2023 구조 변화였다. 따라서 기존 모델의 resolution을 바꾸지 않고, 충분히 큰 부호 전환이 과거 target에서 검출된 game type에만 frozen residual을 적용하는 후보를 채택했다.
