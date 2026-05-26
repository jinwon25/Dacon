# v118 aug+hit — fold0 setupA (2026-05-25T16:54:33)

## 설정
- aug: random yaw [0,2π) + 50% y-flip = **ON**
- band weight: ×2.5 on CV/CA-err ∈ [1cm, 3cm], n=2710
- λ_hit=0.3, τ schedule 0.01→0.003
- setup=A: hidden=64, fc=128, lr=0.0005, p=0.3, wd=0.0001, seed=0
- max_epochs=150, patience=25, batch=256

## 결과 (fold0 va)
- **standalone OOF R-Hit (va): 0.6495**
  - in-band(1-3cm) R-Hit: 0.1651 (n=539)
  - out-band R-Hit: 0.8282
  - fast+turn subset R-Hit (va): 0.2500 (n=20)

## Residual correlation (va fold)
### vs v112_v107_diverse
- corr_x: 0.9798
- corr_y: 0.9817
- corr_z: 0.9771
- corr_3d_mag: 0.9885
- cos_sim_mean: 0.8596
### vs v77 / v90 (참고)
- v77_A_corr_x: 0.9565
- v77_A_corr_y: 0.9537
- v77_A_corr_z: 0.9870
- v77_A_corr_3d_mag: 0.9748
- v90_mirror_corr_x: 0.9563
- v90_mirror_corr_y: 0.9528
- v90_mirror_corr_z: 0.9763
- v90_mirror_corr_3d_mag: 0.9732

## Gate
- OOF ≥ 0.665: ❌ (0.6495)
- corr_3d_mag(v112) < 0.93: ❌ (0.9885)
- **FAIL → STEP 4 skip, v117/v112 마감**