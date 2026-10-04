# 과거 타깃 인코딩의 누출 위험 감사

행 순서·현재 라벨·전체 자료 사전확률에 의존하는 과거 구현의 위험을 확인했습니다. 해당 인코더를 새 중첩 잔차 후보에서 제외한 이유를 기록합니다.

이 문서는 연구 당시의 기록입니다. 최종 결과와 용어·공개 실행 범위는 [문서 안내](../../docs/README.md)를 우선합니다. 아래 수치·판정·명령과 원문은 당시 근거로 보존했습니다.

원제: Target encoding audit

The legacy `PrequentialTargetEncoder` is isolated from all new models. The implementation relies on raw frame order and a full-frame target prior; no exact official chronological key has been established. A row permutation changes its values, a current label can affect future rows in the same group, and changing one label changes the full-frame prior. Therefore it is not used by the nested residual candidates.

```text
                                    audit    value                                                                  interpretation
      row_permutation_sensitivity_max_abs 7.650645                                  nonzero means raw frame order changes features
               current_target_flip_effect 0.000000          the flipped row itself should be zero for strict prequential exclusion
 future_row_effect_of_current_target_flip 0.004975                         future rows in the same group can see the current label
full_frame_global_prior_self_future_shift 0.000050                            fit-frame target mean changes when one label changes
          season_cross_fitted_alternative      NaN requires exact official chronological key; implementation deliberately isolated
```
