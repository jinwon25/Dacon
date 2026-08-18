# Assumption changes

- The formal local `test.csv` contains 5 sample rows; the approximately 245,789-row evaluation batch is server-side and is not reconstructed from test order or distribution.
- The direct adjacent-row as-of check was replaced by a pitcher-group trajectory check. It confirms `asof_pitcher_n` increments by one for 1,473,508 transitions; displayed rate precision gives a 99.935% transition match within 0.01 count units, so integer counts are treated as rounded intervals rather than exact labels.
- Target-free game-block alignment found 7,228 main blocks and 5,980 Trackman games, with 4,810 calendar-bucket assignments but weak distance/placebo separation. It is a candidate generator, not an approved player identity map.
- Full honest nested OOF remains blocked by local runtime. No score or leaderboard claim is derived from a partial run.
