# Legacy state feature specification

This feature definition is intentionally separate from
`src/corrected_top1100_features.py`. The legacy CatBoost axis was fitted with
the earlier endpoint-shift recipe; changing it would create a different model.
Both recipes use official training history, not evaluation-batch statistics.

From `code/`, `src.finalize_legacy_cb_axis` accepts
`--research-project legacy_state`. Its training data location is independently
specified with `--data-project`. Never substitute the corrected feature
definition into the legacy branch merely because its column names overlap.
