# 예측 전 시즌 상태 특징 목록

현재 행과 학습 시점까지의 누적 통계로 만들 수 있는 특징을 정리합니다. 반올림·불가능한 빈도 복원·고정 평활 상수를 검증된 최적 설정으로 해석하지 않습니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: P0-A season-state feature catalog

The audit supports a cumulative, pre-pitch sufficient-statistic interpretation for pitcher state. Candidate features are computed from the current row and train-origin snapshots only:

- pitcher/batter season exposure and season success count;
- career-before-season, prior-season and current-season posterior rates;
- current-versus-career/prior logit deltas and posterior uncertainty;
- newcomer, returner, long-gap, team-switch and role-change flags;
- current-season pitch-mix composition and career drift;
- prev1/3/5 game disagreement against season/career posterior.

Rate displays are rounded before integer reconstruction. If a denominator/rate pair is not within its feasible range, it is excluded from count features and retained only as a missingness diagnostic. Fixed smoothing constants are not treated as validated hyperparameters; later state models must choose discount/process variance on earlier inner folds.
