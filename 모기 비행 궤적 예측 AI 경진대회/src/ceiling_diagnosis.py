"""STEP A — ceiling diagnosis (2026-05-25 D-7 sprint, post v118 FAIL).

3 진단:
  (1) 매끈한 subset (accel_mean 하위 20%)에서 best-possible d 분포
      → CV / CA / poly1 / poly2 / kalman / v112 OOF 비교
      → 중앙값 ~0.007m↑면 "label/observation 노이즈 천장" 결론
  (2) 관측 노이즈 sigma_obs 추정
      - 매끈 subset에서 2차차분 분산: var(Δ²x) ≈ 6 σ²
      - polyfit holdout: 마지막 점을 직전 점들로 재구성한 잔차 std
      → 현 Kalman σ_obs=0.30e-3과 비교
  (3) y_train 좌표 양자화
      - decimal precision, 최소 간격, grid 1e-3 ~ 1e-8 fit 잔차

산출: reports/ceiling_diagnosis.md
"""
from __future__ import annotations
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass
import numpy as np, pandas as pd
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data/cache"
REPORTS = ROOT / "docs/reports"
REPORTS.mkdir(exist_ok=True)
DT = 0.040
DT2 = 2.0 * DT  # +80ms prediction horizon

# ============================================================
# Data
# ============================================================
print("[data] loading X_train, y_train, v112 OOF...")
xc = np.load(CACHE / "xtrain_xtest.npz")
X_train = xc["X_train"].astype(np.float64)        # (10000, 11, 3)
labels = pd.read_csv(ROOT / "data" / "train_labels.csv")
ids_sorted = [f"TRAIN_{i:05d}" for i in range(1, 10001)]
y_train = labels.set_index("id").loc[ids_sorted][["x", "y", "z"]].values.astype(np.float64)

oof_v112 = np.load(CACHE / "v112_v107_diverse_weights.npz")["oof_pred"].astype(np.float64)
kalman_pred_train = None
try:
    kc = np.load(CACHE / "kalman.npz")
    # try standard keys
    for k in ("pred_train", "kalman_train", "y_train_pred", "train_pred"):
        if k in kc.files:
            kalman_pred_train = kc[k].astype(np.float64); break
    if kalman_pred_train is None:
        print(f"[kalman] keys = {kc.files}")
except FileNotFoundError:
    pass

# ============================================================
# 기하학적 baseline 예측들 (전 샘플)
# ============================================================
print("[predictors] building geometric baselines...")
v_last = (X_train[:, -1] - X_train[:, -2]) / DT
v_prev = (X_train[:, -2] - X_train[:, -3]) / DT
a_last = (v_last - v_prev) / DT

cv_pred = X_train[:, -1] + v_last * DT2
ca_pred = X_train[:, -1] + v_last * DT2 + 0.5 * a_last * (DT2 ** 2)

def polyfit_predict(X, order=2, last=6, t_ahead=DT2):
    """per-axis polyfit on last `last` samples (t=-last+1..0 in units of DT), eval at +t_ahead."""
    t = np.arange(-last + 1, 1) * DT
    A = np.vander(t, order + 1, increasing=True)
    pinv = np.linalg.pinv(A)             # (order+1, last)
    eval_vec = np.array([t_ahead ** k for k in range(order + 1)])  # (order+1,)
    N = X.shape[0]
    out = np.zeros((N, 3))
    for j in range(3):
        Yj = X[:, -last:, j]              # (N, last)
        coefs = (pinv @ Yj.T).T           # (N, order+1)
        out[:, j] = coefs @ eval_vec
    return out

poly1_pred = polyfit_predict(X_train, order=1, last=6)
poly2_pred = polyfit_predict(X_train, order=2, last=6)
poly2_l4 = polyfit_predict(X_train, order=2, last=4)
poly3_l6 = polyfit_predict(X_train, order=3, last=6)

# ============================================================
# accel_mean → 매끈 subset 정의
# ============================================================
print("[masks] computing accel_mean and smooth subset...")
disp = np.diff(X_train, axis=1)                # (10000, 10, 3)
vel  = disp / DT                                # m/s
acc  = np.diff(vel, axis=1) / DT                # (10000, 9, 3)  m/s²
accel_mean = np.linalg.norm(acc, axis=-1).mean(axis=1)  # (10000,)

q20 = np.quantile(accel_mean, 0.20)
q50 = np.quantile(accel_mean, 0.50)
q80 = np.quantile(accel_mean, 0.80)
smooth_mask = accel_mean <= q20             # 매끈 하위 20% (≈ 2000)
rough_mask  = accel_mean >= q80             # 거친 상위 20%
print(f"  accel_mean quantiles: q20={q20:.3f}  q50={q50:.3f}  q80={q80:.3f}  max={accel_mean.max():.2f} m/s²")

# fast+turn (메모리 hard subset 정의 일치)
sp_last = np.linalg.norm(v_last, axis=-1)
v_prev_mean = vel[:, :-1, :].mean(axis=1)
na = np.linalg.norm(v_last, axis=-1); nb = np.linalg.norm(v_prev_mean, axis=-1)
turn_cos = np.clip((v_last * v_prev_mean).sum(-1) / np.maximum(na * nb, 1e-12), -1, 1)
hard_mask = (sp_last > 1.0) & (turn_cos < 0.5)
print(f"  smooth20 n={smooth_mask.sum()}  rough20 n={rough_mask.sum()}  fast+turn n={hard_mask.sum()}")

# ============================================================
# 예측 d 통계
# ============================================================
def d_of(pred): return np.linalg.norm(pred - y_train, axis=-1)
def stats(d):
    return {
        "n": int(len(d)),
        "mean_mm": float(d.mean()*1000),
        "median_mm": float(np.median(d)*1000),
        "p25_mm": float(np.quantile(d, 0.25)*1000),
        "p75_mm": float(np.quantile(d, 0.75)*1000),
        "p90_mm": float(np.quantile(d, 0.90)*1000),
        "hit1cm": float((d <= 0.01).mean()),
    }

preds = {
    "CV":      cv_pred,
    "CA":      ca_pred,
    "poly1_6": poly1_pred,
    "poly2_6": poly2_pred,
    "poly2_4": poly2_l4,
    "poly3_6": poly3_l6,
    "v112OOF": oof_v112,
}
if kalman_pred_train is not None: preds["kalman"] = kalman_pred_train

subsets = {"all": np.ones(10000, dtype=bool), "smooth20": smooth_mask, "rough20": rough_mask, "fast+turn": hard_mask}

print("\n[STEP A.1] best-possible d on smooth subset")
results_d = {}
for sn, sm in subsets.items():
    row = {}
    for pn, pp in preds.items():
        d = d_of(pp)[sm]
        row[pn] = stats(d)
    results_d[sn] = row

# Find best predictor per subset by median
print(f"  {'subset':<10} | {'predictor':<10} | median_mm | hit1cm")
for sn, sm in subsets.items():
    best_p = min(results_d[sn].items(), key=lambda kv: kv[1]["median_mm"])
    bn, bs = best_p
    print(f"  {sn:<10} | {bn:<10} | {bs['median_mm']:>8.3f}  | {bs['hit1cm']:.3f}")

# Oracle envelope: per-sample minimum over geometric predictors → 어떤 단순 외삽도 못 막은 floor
geom_names = ["CV", "CA", "poly1_6", "poly2_6", "poly2_4", "poly3_6"]
geom_d = np.stack([d_of(preds[n]) for n in geom_names], axis=1)  # (10000, 6)
oracle_d = geom_d.min(axis=1)
print("\n[STEP A.1b] oracle (per-sample min over 6 geom predictors) — '데이터가 허락하는 best 단순 외삽 floor'")
for sn, sm in subsets.items():
    print(f"  {sn:<10}: median={np.median(oracle_d[sm])*1000:.3f}mm  hit1cm={(oracle_d[sm]<=0.01).mean():.3f}")

# ============================================================
# STEP A.2 — 관측 노이즈 σ 추정
# ============================================================
print("\n[STEP A.2] observation noise σ estimation")

# (a) 2nd-difference on smoothest subset: var(Δ²x) ≈ 6 σ_obs²  (assume true accel ≈ 0)
sd = np.diff(np.diff(X_train, axis=1), axis=1)  # (10000, 9, 3) m
sd_smooth = sd[smooth_mask].reshape(-1, 3)
sigma_sd = np.sqrt(sd_smooth.var(axis=0) / 6.0) * 1000   # mm
sigma_sd_3d = np.sqrt(sigma_sd**2).sum()**0.5

# (b) holdout poly2: 마지막 점을 직전 6 점으로 polyfit → 잔차 std
def holdout_sigma(X, order=2, ctx=6, mask=None):
    t_ctx = np.arange(-ctx, 0) * DT       # context indices: t=-ctx*DT..t=-DT
    A = np.vander(t_ctx, order + 1, increasing=True)
    pinv = np.linalg.pinv(A)
    eval_vec = np.array([0.0 ** k for k in range(order + 1)])  # eval at t=0 (last point)
    eval_vec[0] = 1.0
    N = X.shape[0]
    pred = np.zeros((N, 3))
    for j in range(3):
        Yj = X[:, -ctx-1:-1, j]
        coefs = (pinv @ Yj.T).T
        pred[:, j] = coefs @ eval_vec
    resid = X[:, -1, :] - pred   # (N, 3)
    if mask is not None: resid = resid[mask]
    return resid

resid_holdout = holdout_sigma(X_train, order=2, ctx=6, mask=smooth_mask)
sigma_holdout = resid_holdout.std(axis=0) * 1000

# (c) reference: Kalman code's hardcoded σ_obs
sigma_kalman_mm = 0.30e-3 * 1000   # = 0.30 mm

print(f"  (a) 2nd-diff on smooth20:        σ = {sigma_sd[0]:.3f} / {sigma_sd[1]:.3f} / {sigma_sd[2]:.3f} mm  (per axis x/y/z)")
print(f"  (b) poly2 holdout last on smooth20: σ = {sigma_holdout[0]:.3f} / {sigma_holdout[1]:.3f} / {sigma_holdout[2]:.3f} mm")
print(f"  (c) Kalman code 가정:            σ = {sigma_kalman_mm:.3f} mm (per axis)")

# Implied: at +80ms (2 steps ahead), pure observation noise propagates: σ_pred ≈ σ_obs (since the +80ms target is also observed with same noise)
# Even oracle predictor cannot do better than σ_obs on the target observation itself.
# 3D distance from noise alone ≈ sqrt(3) * σ_obs (Rayleigh-ish median ≈ 1.538 σ for 3D)
sigma_avg_holdout = sigma_holdout.mean()
implied_floor_median = 1.538 * sigma_avg_holdout
print(f"  → 관측노이즈만으로 d 중앙값 floor ≈ 1.538 × {sigma_avg_holdout:.3f}mm = {implied_floor_median:.3f}mm")

# ============================================================
# STEP A.3 — y_train 양자화
# ============================================================
print("\n[STEP A.3] target quantization signature")
quant_report = {}
for j, lab in enumerate("xyz"):
    yj = y_train[:, j]
    # Detect grid: try g = 10^-k, check max integer-rounding residual
    grid_fit = {}
    for k in range(2, 9):
        g = 10.0 ** (-k)
        r = yj / g - np.round(yj / g)
        grid_fit[f"1e-{k}"] = float(np.abs(r).max())
    # Unique value gaps
    sy = np.sort(np.unique(np.round(yj, 8)))
    gaps = np.diff(sy)
    quant_report[lab] = {
        "n_unique": int(len(sy)),
        "min_gap": float(gaps.min()),
        "median_gap": float(np.median(gaps)),
        "max_gap": float(gaps.max()),
        "grid_fit_max_resid": grid_fit,
        "range": [float(yj.min()), float(yj.max())],
    }
    print(f"  {lab}: n_unique={len(sy):>6d}  min_gap={gaps.min():.2e}  median_gap={np.median(gaps):.2e}")
    print(f"      grid resid max @ 1e-3: {grid_fit['1e-3']:.4f}, 1e-4: {grid_fit['1e-4']:.4f}, 1e-5: {grid_fit['1e-5']:.4f}, 1e-6: {grid_fit['1e-6']:.4f}")

# ============================================================
# 결론 자동판정 & 보고서
# ============================================================
smooth_v112 = stats(d_of(oof_v112)[smooth_mask])
smooth_oracle_med_mm = np.median(oracle_d[smooth_mask]) * 1000
ceiling_call = "label/관측노이즈 천장 가능성 큼" if smooth_oracle_med_mm >= 7.0 else "model headroom 남음 (oracle 매끈<7mm)"

md_lines = []
md_lines.append("# STEP A — Ceiling Diagnosis 결과")
md_lines.append("")
md_lines.append(f"날짜: 2026-05-25 (D-7 sprint, v118 FAIL 직후 ceiling 진단)")
md_lines.append("")
md_lines.append("## 0. 정의")
md_lines.append("- 매끈 subset(smooth20) = accel_mean 하위 20% (~2000 샘플)")
md_lines.append(f"- accel_mean = 각 샘플의 9개 timestep 가속도 norm 평균 (m/s²)")
md_lines.append(f"- accel_mean quantile: q20={q20:.3f}, q50={q50:.3f}, q80={q80:.3f} m/s²")
md_lines.append(f"- fast+turn subset (hard) = sp_last>1.0 AND turn_cos<0.5,  n={int(hard_mask.sum())}")
md_lines.append("")

# ====== STEP A.1 표 ======
md_lines.append("## A.1 매끈 subset best-possible d (단위 mm)")
md_lines.append("")
md_lines.append("| subset | predictor | n | median | mean | p25 | p75 | p90 | hit1cm |")
md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for sn in ["smooth20", "all", "rough20", "fast+turn"]:
    for pn, pp in preds.items():
        d = d_of(pp)[subsets[sn]]
        s = stats(d)
        md_lines.append(f"| {sn} | {pn} | {s['n']} | {s['median_mm']:.3f} | {s['mean_mm']:.3f} | {s['p25_mm']:.3f} | {s['p75_mm']:.3f} | {s['p90_mm']:.3f} | {s['hit1cm']:.3f} |")
md_lines.append("")

# oracle envelope
md_lines.append("### A.1b oracle envelope (per-sample 최소 over 6 geom predictors)")
md_lines.append("'어떤 단순 외삽 (CV/CA/poly1~3)도 못 막은 잔차' = label/관측노이즈 + saccade 비결정성 lower bound")
md_lines.append("")
md_lines.append("| subset | median_mm | mean_mm | hit1cm |")
md_lines.append("|---|---:|---:|---:|")
for sn in ["smooth20", "all", "rough20", "fast+turn"]:
    d = oracle_d[subsets[sn]]
    md_lines.append(f"| {sn} | {np.median(d)*1000:.3f} | {d.mean()*1000:.3f} | {(d<=0.01).mean():.3f} |")
md_lines.append("")

# ====== STEP A.2 ======
md_lines.append("## A.2 관측 노이즈 σ")
md_lines.append("")
md_lines.append("| 방법 | σ_x (mm) | σ_y (mm) | σ_z (mm) | 비고 |")
md_lines.append("|---|---:|---:|---:|---|")
md_lines.append(f"| (a) 2차차분 var (smooth20, true accel≈0) | {sigma_sd[0]:.3f} | {sigma_sd[1]:.3f} | {sigma_sd[2]:.3f} | σ = sqrt(var/6) |")
md_lines.append(f"| (b) poly2 holdout last (smooth20) | {sigma_holdout[0]:.3f} | {sigma_holdout[1]:.3f} | {sigma_holdout[2]:.3f} | 마지막 점 직전6점 폴리피팅 잔차 |")
md_lines.append(f"| (c) Kalman code 가정 | {sigma_kalman_mm:.3f} | {sigma_kalman_mm:.3f} | {sigma_kalman_mm:.3f} | sigma_obs=0.30e-3 |")
md_lines.append("")
md_lines.append(f"**관측노이즈만으로 d 중앙값 floor (3D)**: ≈ 1.538 × σ_avg = **{implied_floor_median:.3f} mm**")
md_lines.append("")
md_lines.append(f"→ Kalman σ={sigma_kalman_mm:.3f}mm 가정 vs 실측 σ≈{sigma_holdout.mean():.3f}mm: **{'심한 under-trust' if sigma_holdout.mean() > 2*sigma_kalman_mm else '근사 일치'}**")
md_lines.append("")

# ====== STEP A.3 ======
md_lines.append("## A.3 target 좌표 양자화")
md_lines.append("")
md_lines.append("| axis | n_unique | min_gap | median_gap | range |")
md_lines.append("|---|---:|---:|---:|---|")
for lab in "xyz":
    q = quant_report[lab]
    md_lines.append(f"| {lab} | {q['n_unique']} | {q['min_gap']:.2e} | {q['median_gap']:.2e} | [{q['range'][0]:.3f}, {q['range'][1]:.3f}] |")
md_lines.append("")
md_lines.append("Grid fit max-residual (작을수록 강한 grid signature):")
md_lines.append("")
md_lines.append("| axis | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |")
md_lines.append("|---|---:|---:|---:|---:|---:|")
for lab in "xyz":
    g = quant_report[lab]["grid_fit_max_resid"]
    md_lines.append(f"| {lab} | {g['1e-3']:.4f} | {g['1e-4']:.4f} | {g['1e-5']:.4f} | {g['1e-6']:.4f} | {g['1e-7']:.4f} |")
md_lines.append("")

# ====== 결론 ======
md_lines.append("## 결론 / 분기 판정")
md_lines.append("")
md_lines.append(f"**A.1 매끈subset oracle d 중앙값 = {smooth_oracle_med_mm:.3f} mm**")
md_lines.append(f"  → {ceiling_call}")
md_lines.append("")
md_lines.append(f"**A.1 매끈subset v112 d 중앙값 = {smooth_v112['median_mm']:.3f} mm  (v112가 smooth에서 oracle 대비 얼마나 거리)**")
md_lines.append("")
v112_full = stats(d_of(oof_v112))
md_lines.append(f"**전체 v112 OOF hit1cm = {v112_full['hit1cm']:.4f}** (R-Hit; v112 OOF 0.6768과 비교용)")
md_lines.append("")

# 데이터-driven STEP B/C/D 추천
recs = []
if smooth_oracle_med_mm >= 7.0:
    recs.append("STEP A.1 → **천장 시그널**: 매끈 subset oracle median ≥ 7mm. STEP D(aWTA) 기대 lift 약. STEP B만 빠르게 시도하고 v112 마감 권고.")
else:
    recs.append("STEP A.1 → 매끈 subset oracle median < 7mm: 모델에 여전히 headroom. STEP D 시도 가치 있음.")

x_grid_resid_4 = quant_report["x"]["grid_fit_max_resid"]["1e-4"]
x_grid_resid_5 = quant_report["x"]["grid_fit_max_resid"]["1e-5"]
if x_grid_resid_5 < 0.01:
    recs.append(f"STEP A.3 → x축 1e-5 grid 잔차 {x_grid_resid_5:.4f} < 0.01 = **양자화 signature 발견**. STEP C(snapping) 실행 가치.")
elif x_grid_resid_4 < 0.01:
    recs.append(f"STEP A.3 → x축 1e-4 grid 잔차 {x_grid_resid_4:.4f} < 0.01 = 약한 grid signature.")
else:
    recs.append("STEP A.3 → grid signature 없음. STEP C(snapping) 무효 — skip.")

if sigma_holdout.mean() > 2 * sigma_kalman_mm:
    recs.append(f"STEP A.2 → 실측 σ={sigma_holdout.mean():.3f}mm ≫ Kalman 가정 {sigma_kalman_mm:.3f}mm. Kalman re-tune 가치 (1h). 단 v112 안에 이미 다양 σ Kalman 섞여있으면 미미.")

for r in recs:
    md_lines.append(f"- {r}")
md_lines.append("")

(REPORTS / "ceiling_diagnosis.md").write_text("\n".join(md_lines), encoding="utf-8")
print(f"\n[done] reports/ceiling_diagnosis.md 저장 ({len(md_lines)} lines)")

# JSON dump for downstream STEPs
out = {
    "smooth_oracle_median_mm": smooth_oracle_med_mm,
    "smooth_v112_median_mm":   smooth_v112["median_mm"],
    "sigma_obs_holdout_mm":    sigma_holdout.tolist(),
    "sigma_obs_2nd_diff_mm":   sigma_sd.tolist(),
    "quant_report":            quant_report,
    "ceiling_call":            ceiling_call,
    "v112_full_hit1cm":        v112_full["hit1cm"],
    "v112_smooth_hit1cm":      smooth_v112["hit1cm"],
    "smooth_thr_accel_ms2":    float(q20),
    "hard_n":                  int(hard_mask.sum()),
    "smooth_n":                int(smooth_mask.sum()),
}
(REPORTS / "ceiling_diagnosis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"[done] reports/ceiling_diagnosis.json 저장")
