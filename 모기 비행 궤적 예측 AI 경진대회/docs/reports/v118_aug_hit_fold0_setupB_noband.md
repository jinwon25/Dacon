# v118 aug+hit — fold0 setupB (2026-05-25T17:08:19)

## 설정
- aug: random yaw [0,2π) + 50% y-flip = **ON**
- band weight: ×2.5 on CV/CA-err ∈ [1cm, 3cm], n=0
- λ_hit=0.3, τ schedule 0.01→0.003
- setup=B: hidden=64, fc=128, lr=0.001, p=0.1, wd=0.0001, seed=0
- max_epochs=150, patience=25, batch=256

## 결과 (fold0 va)
- **standalone OOF R-Hit (va): 0.6715**
  - in-band(1-3cm) R-Hit: 0.0000 (n=0)
  - out-band R-Hit: 0.6715
  - fast+turn subset R-Hit (va): 0.2500 (n=20)

## Residual correlation (va fold)
### vs v112_v107_diverse
- corr_x: 0.9870
- corr_y: 0.9853
- corr_z: 0.9834
- corr_3d_mag: 0.9922
- cos_sim_mean: 0.8901
### vs v77 / v90 (참고)
- v77_A_corr_x: 0.9536
- v77_A_corr_y: 0.9492
- v77_A_corr_z: 0.9881
- v77_A_corr_3d_mag: 0.9736
- v90_mirror_corr_x: 0.9545
- v90_mirror_corr_y: 0.9489
- v90_mirror_corr_z: 0.9855
- v90_mirror_corr_3d_mag: 0.9727

## Gate
- OOF ≥ 0.665: ✅ (0.6715)
- corr_3d_mag(v112) < 0.93: ❌ (0.9922)
- **FAIL → STEP 4 skip, v117/v112 마감**