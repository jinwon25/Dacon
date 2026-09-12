# P0-B target-free structural alignment audit

The aligner uses only season/month/day-of-week, legal count/out/inning transitions, game length, half distribution and hand tokens. `control_success`, target encodings and target-derived statistics are prohibited by an explicit whitelist assertion. Main does not contain game_date or game_id, so this is a game-candidate structural alignment rather than a final pitch-level identity proof.

Real candidate summary:

```json
{
  "n_main_games": 7228,
  "n_trackman_games": 5980,
  "n_matches": 4810,
  "high_rate": 0.0681912681912682,
  "median_distance": 1.9261443743603022,
  "median_margin": 0.09509072217137471
}
```

Placebo cyclic shifts are written to `alignment_placebo_fdr.csv`. Existing annual player linkage is not promoted to a new physical profile solely from this audit; high-confidence structural coverage and permutation separation must be confirmed first. Medium/low/unmatched rows therefore remain on the V2 fallback.
