# 다음 세션 브리프 (2026-05-26 v122c LB 0.6912 돌파 직후)

## LB 현황 — plateau 0.6888 돌파 확정

| 모델 | OOF | LB | 변환률 | 제출 |
|---|---:|---:|---:|---|
| v106 DE15w | 0.6770 | 0.6888 | +0.0118 | 2026-05-24 |
| v112_v107_diverse | 0.6768 | 0.6888 | +0.0120 | 2026-05-25 |
| v110_v3 29w | 0.6775 | 0.6884 | +0.0109 (over-fit) | 2026-05-25 |
| **v122c_v121diverse (Neural ODE blend)** | **0.6769** | **0.6912** ★ | **+0.0143** ★ | **2026-05-26 16:05** |

**핵심 결론**: Neural ODE paradigm (v120 family) 도입으로 plateau 절대 돌파. **변환률 +0.0143**이 최고. corr_3d <0.93 게이트 폐기 확정 (L2 distance가 진짜 paradigm 지표).

## EDA 핵심 발견 (2026-05-26, reports/eda_post_v122c.md)

### Oracle bound
- v112 hit 0.6768, v120 hit 0.6610
- **oracle min(v112, v120) = 0.6989** (per-sample 최적 선택의 천장)
- v122c가 oracle hit의 96.72% 캡처 → selector 추가 lift 잠재 = +0.022

### Selector 카드 dead (negative result)
- v125_disagreement_selector (22 features + |v112-v120| + MLP 5-fold 3-seed) → OOF AUC = **0.5562**
- selector accuracy 시뮬: 55% → 손해(0.6715), 65% → break-even, 70%+에서 +0.002 lift만
- only_v112 vs only_v120 메타특징 식별력 Δz < 0.14 → 단일 paradigm pair selector 불가능
- **dead. paradigm 추가가 진짜 카드.**

### Boundary subset (v122c 0.8-1.5cm miss, n=2535)
- v112 hit 40.47%, v120 hit 36.29%, oracle 49.15% (+8.68%p)
- 여기는 paradigm 더 추가 시 lift 가능 영역

### Neither subset (n=3011, 둘 다 hit 못 함)
- mean distance 26mm → 새 paradigm 진짜 필요한 영역

### Train vs Test 분포
- speed_last: train 0.639 vs test 0.606 (test가 살짝 slow)
- ks proxy 0.001~0.104 → 분포 차이 미세. **TTA 거의 효과 없음 예상**.

## 다음 우선순위 카드

| 우선 | 카드 | 시간 | 잠재 LB lift |
|---|---|---|---|
| **1** | **v120 n_steps=2/4 multi-step RK4** | CPU 30min/Colab 10min | +0.001~+0.004 |
| **2** | **v120 latent=128/hidden=128 big** | Colab 15min | +0.001~+0.003 |
| **3** | **v126 FFT feature (rfft mag/phase)** | Colab 15min | +0.001~+0.004 |
| 4 | Neural CDE (torchcde, kidger 2020) | 1-2일 | +0.002~+0.005 |
| 5 | Frenet-frame coordinate transformation | 1-2일 | +0.001~+0.003 |
| 6 | per-axis (x/y/z 분리) v120 변종 | 1일 | +0.001~+0.003 |
| ❌ | meta-only selector (v125) | - | dead, AUC 0.5562 |

## 추천 sprint plan

**Plan A — 카드 1+2+3 (1-2일, Colab T4)**:
1. v120 n_steps=2 full (mode=full) → cache/v120_n2_full_state.npz
2. v120 latent=128 full → cache/v120_big_full_state.npz  
3. v126 FFT full → cache/v126_full_state.npz
4. v110_de_ensemble.py pool에 추가 (8 멤버 → 11 멤버) → DE 재계산 → v122d
5. 예상 OOF 0.6790~0.6810, LB 변환 +0.0143 → **0.6933~0.6953**

**Plan B — Neural CDE (1-2일)**:
1. `pip install torchcde` 
2. cubic Hermite interpolation for 11 obs → CDE backbone
3. 같은 mirror aug, 같은 metric으로 학습
4. Pool 추가 → 같은 DE blend

## 코드 자산

- `scripts/v120_neural_ode.py` — base ODE model (이제 `--n_steps/--latent_dim/--hidden` CLI)
- `scripts/v126_fft_neural_ode.py` — FFT 변종 (NEW)
- `scripts/v125_disagreement_selector.py` — selector (dead, 보존만)
- `scripts/eda_post_v122c.py` — oracle analysis (재실행 가능)
- `scripts/eda_selector_probe.py` — only_v112 vs only_v120 진단
- `cache/v122c_v121diverse_weights.npz` — 현재 LB 0.6912 best blend

## 핵심 cache 보존

- `v107_state, v107_setupB_state` — pool 멤버
- `v108_* (7 variants)` — boundary per-axis
- `v109_K4/K8_state, _pool` — MDN/WTA
- `v110_*_weights, v111_*, v112_*_weights, v115_*` — 모든 blend/stacker
- `v118_*` — STEP3 residual corr 진단 (보존 명시)
- `v120_full_state, v121_cap10/15, v122c_v121diverse_weights` — Neural ODE family
- `kalman, geomedian, kalmans_multi, meta_features, noise_*, phys_state, xtrain_xtest` — 공유 베이스

## 정리된 것 (2026-05-26)
- scripts/__pycache__/
- open/_archive_submissions/ (68M, 115 files)
- archive/legacy_versions/, archive/v10/v13/v16 (옛 cache)
- logs/ (옛 텍스트 로그)
- outputs/01_best_public, 02_boundary_oof, 03~08 (5 subfolders + 90_archive + selector_full) → 161M → 5M

## 사용자 메모 (2026-05-26)
- D-day 마감일 = 2026-06-01 10:00 (5일 남음)
- Colab T4 + VS Code 원격 커널 환경
- "쓸데없는 파일도 지워주고" — 정리 완료
- "시니어 모델러로 진행" 자율 모드
