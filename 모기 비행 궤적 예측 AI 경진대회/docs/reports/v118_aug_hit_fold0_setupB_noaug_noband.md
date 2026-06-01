# v118 aug+hit — fold0 setupB (2026-05-25T17:01:14)

## 설정
- aug: random yaw [0,2π) + 50% y-flip = **OFF**
- band weight: ×2.5 on CV/CA-err ∈ [1cm, 3cm], n=0
- λ_hit=0.3, τ schedule 0.01→0.003
- setup=B: hidden=64, fc=128, lr=0.001, p=0.1, wd=0.0001, seed=0
- max_epochs=150, patience=25, batch=256

## 결과 (fold0 va)
- **standalone OOF R-Hit (va): 0.6655**
  - in-band(1-3cm) R-Hit: 0.0000 (n=0)
  - out-band R-Hit: 0.6655
  - fast+turn subset R-Hit (va): 0.3500 (n=20)

## Residual correlation (va fold)
### vs v112_v107_diverse
- corr_x: 0.9872
- corr_y: 0.9841
- corr_z: 0.9805
- corr_3d_mag: 0.9919
- cos_sim_mean: 0.8995
### vs v77 / v90 (참고)
- v77_A_corr_x: 0.9409
- v77_A_corr_y: 0.9406
- v77_A_corr_z: 0.9923
- v77_A_corr_3d_mag: 0.9708
- v90_mirror_corr_x: 0.9410
- v90_mirror_corr_y: 0.9385
- v90_mirror_corr_z: 0.9856
- v90_mirror_corr_3d_mag: 0.9693

## Gate
- OOF ≥ 0.665: ✅ (0.6655)
- corr_3d_mag(v112) < 0.93: ❌ (0.9919)
- **FAIL → STEP 4 skip, v117/v112 마감**