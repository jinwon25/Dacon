# Target encoding audit

The legacy `PrequentialTargetEncoder` is isolated from all new models. The implementation relies on raw frame order and a full-frame target prior; no exact official chronological key has been established. A row permutation changes its values, a current label can affect future rows in the same group, and changing one label changes the full-frame prior. Therefore it is not used by the nested residual candidates.

```text
                                    audit    value                                                                  interpretation
      row_permutation_sensitivity_max_abs 7.650645                                  nonzero means raw frame order changes features
               current_target_flip_effect 0.000000          the flipped row itself should be zero for strict prequential exclusion
 future_row_effect_of_current_target_flip 0.004975                         future rows in the same group can see the current label
full_frame_global_prior_self_future_shift 0.000050                            fit-frame target mean changes when one label changes
          season_cross_fitted_alternative      NaN requires exact official chronological key; implementation deliberately isolated
```
