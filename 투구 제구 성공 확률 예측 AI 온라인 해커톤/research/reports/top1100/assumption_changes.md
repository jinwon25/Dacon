# 초기 연구의 가정 수정

공식 테스트 예제와 서버 평가 자료, 반올림된 누적 통계와 정확한 정답, 구조 연결 후보와 신원 확정을 구분한 기록입니다. 부분 중첩 실행으로 점수나 학습 외 예측을 주장하지 않았습니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: Assumption changes

- The formal local `test.csv` contains 5 sample rows; the approximately 245,789-row evaluation batch is server-side and is not reconstructed from test order or distribution.
- The direct adjacent-row as-of check was replaced by a pitcher-group trajectory check. It confirms `asof_pitcher_n` increments by one for 1,473,508 transitions; displayed rate precision gives a 99.935% transition match within 0.01 count units, so integer counts are treated as rounded intervals rather than exact labels.
- Target-free game-block alignment found 7,228 main blocks and 5,980 Trackman games, with 4,810 calendar-bucket assignments but weak distance/placebo separation. It is a candidate generator, not an approved player identity map.
- Full honest nested OOF remains blocked by local runtime. No score or leaderboard claim is derived from a partial run.
