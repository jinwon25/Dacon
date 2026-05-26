# Base 입력표현 audit (STEP 1)

날짜: 2026-05-25
대상: 현 pool 주력 NN base (v23 framework 상속군)

## (a) 변위 vs 절대좌표 타깃 — 변위 ✅

`scripts/v23_train.py` L489:
```python
target_T8 = rotate_xy(y_train - kalman_train, theta_train)
target_F  = rotate_xy(y_train - X_train[:, -1], theta_train)
target_W  = rotate_xy(y_train - kalman_train_alt, theta_train)
```
- main target = **Kalman residual displacement** (y - kalman_pred), canonical frame
- aux F = relative to last position
- aux W = relative to alt-Kalman (σ_obs=1.0e-3 변종)
- 출력은 `tanh * 2cm`로 ±2cm clamp (v23 GRUModelMultiAux L323)

→ **절대좌표가 아니라 잔차 displacement** 학습. 이미 채택.

## (b) Canonical frame (속도벡터 정렬) ✅

L486-488:
```python
v_last_train = (X_train[:, -1] - X_train[:, -2]) / DT
theta_train  = yaw_angle(v_last_train)   # arctan2(vy, vx)
target_T8    = rotate_xy(y_train - kalman_train, theta_train)
```
- yaw = **마지막 1 step velocity** 방향 → +x로 회전
- target/aux/test 모두 같은 θ 사용
- 추론 시 `inverse_rotate_xy`로 역회전 (L432)

→ **마지막 구간 속도벡터 정렬 canonical 이미 사용**. 사용자 STEP1 게이트 만족.

## (c) 정규화 방식

v23 L355-362:
```python
sc_seq  = StandardScaler().fit(seq_arr[tr].reshape(-1, seq_arr.shape[2]))
sc_scal = StandardScaler().fit(scal_arr[tr])
```
- **per-fold**, train fold에서만 fit
- seq는 (N·T, C)로 reshape해서 채널별 standardization
- scal은 row별 standardization
- log1p 사전 적용 (scal: 15개 컬럼)

## (d) 증강 — random yaw 풀 증강 미시도 ⚠

| script | aug |
|---|---|
| v23, v77 (BiGRU) | 없음 |
| v90 (yaw_mirror_aug) | y-mirror (좌우 대칭만, 2x) |
| v96 (yaw_4view_aug) | 4 fixed: normal + mirror + yaw±20° |
| v107 (deep Transformer) | 없음 (v23 framework 그대로) |

**미시도**:
- random yaw [0, 2π) 풀 증강 (epoch마다)
- input jitter (관측 노이즈 σ 추정)
- hit-aware sample weight (1-3cm 밴드 ×2~3)
- soft_hit τ-schedule (0.01 → 0.003)

## 결론

> STEP1 게이트(canonical+변위 이미 사용) → **PASS, STEP 3로 직행**.
>
> 미시도된 (random yaw 풀, y-flip 50%, hit-aware loss + 1-3cm 밴드 weighting)을 합쳐서 BiGRU 1개 single-fold 학습한다. 게이트: standalone OOF ≥ 0.665 AND residual 상관 < 0.93.
