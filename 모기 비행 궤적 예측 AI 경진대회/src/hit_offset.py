"""STEP B — hit-aware offset (global 3D + velocity-aligned body frame).

목표: v112 OOF 위에 작은 δ를 더해 R-Hit@1cm 극대화.
gate: OOF lift +0.0008↑일 때만 test_pred에 적용 (작아도 노이즈상회 8샘플flip).

3 offset 탐색:
  (1) global 3D δ — coarse → fine grid 검색
  (2) global body-frame δ — 마지막 속도 정렬 (along/cross_h/vertical), per-sample 회전 후 적용
  (3) speed-conditional global δ — 속도 bin별 다른 δ (overfit 우려 → 따로 보고만)

산출: reports/hit_offset.md, final_candidates/submission_v112_offset.csv (gate 통과 시만)
"""
from __future__ import annotations
import sys
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except: pass
import numpy as np, pandas as pd
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data/cache"; REPORTS = ROOT / "docs/reports"; FINAL = ROOT / "submissions/historical"
DT = 0.040

# data
xc = np.load(CACHE / "xtrain_xtest.npz")
X_train = xc["X_train"]; X_test = xc["X_test"]
labels = pd.read_csv(ROOT / "data" / "train_labels.csv")
ids_sorted_train = [f"TRAIN_{i:05d}" for i in range(1, 10001)]
y_train = labels.set_index("id").loc[ids_sorted_train][["x","y","z"]].values.astype(np.float64)

w112 = np.load(CACHE / "v112_v107_diverse_weights.npz")
oof = w112["oof_pred"].astype(np.float64)        # (10000, 3)
test_pred = w112["test_pred"].astype(np.float64) # (10000, 3)

base_d = np.linalg.norm(oof - y_train, axis=-1)
base_hit = (base_d <= 0.01).mean()
print(f"[base] v112 OOF hit1cm = {base_hit:.4f}  (median d = {np.median(base_d)*1000:.3f} mm)")

# residual stats — global bias clue
resid = oof - y_train
print(f"[residual] mean = {resid.mean(axis=0)*1000} mm")
print(f"[residual] std  = {resid.std(axis=0)*1000} mm")

# ============================================================
# (1) Global 3D δ — two-stage grid
# ============================================================
def hit_with_offset(pred, y, delta):
    d = np.linalg.norm(pred + delta - y, axis=-1)
    return (d <= 0.01).mean()

print("\n[B.1] Global 3D δ grid search (coarse → fine)")
# coarse: ±3mm step 0.5mm → 13^3 = 2197
best = (base_hit, np.array([0.0,0.0,0.0]))
coarse_range = np.arange(-3, 3.01, 0.5) * 1e-3
print(f"  coarse 13^3={len(coarse_range)**3} eval, base = {base_hit:.4f}")
for dx in coarse_range:
    for dy in coarse_range:
        for dz in coarse_range:
            h = hit_with_offset(oof, y_train, np.array([dx,dy,dz]))
            if h > best[0]: best = (h, np.array([dx,dy,dz]))
print(f"  coarse best: δ = ({best[1][0]*1000:+.2f}, {best[1][1]*1000:+.2f}, {best[1][2]*1000:+.2f}) mm  hit = {best[0]:.4f}  Δ = {best[0]-base_hit:+.4f}")

# fine: around coarse best ±0.5mm step 0.1mm
fine_range = np.arange(-0.5, 0.51, 0.1) * 1e-3
c = best[1]; best_fine = best
for dx in c[0] + fine_range:
    for dy in c[1] + fine_range:
        for dz in c[2] + fine_range:
            h = hit_with_offset(oof, y_train, np.array([dx,dy,dz]))
            if h > best_fine[0]: best_fine = (h, np.array([dx,dy,dz]))
hit_global, delta_global = best_fine
print(f"  fine best:   δ = ({delta_global[0]*1000:+.3f}, {delta_global[1]*1000:+.3f}, {delta_global[2]*1000:+.3f}) mm  hit = {hit_global:.4f}  Δ = {hit_global-base_hit:+.4f}")

# ============================================================
# (2) Velocity-aligned body-frame δ
# ============================================================
# Body frame per sample:
#   e1 = horizontal unit along v_last (project to xy plane)
#   e2 = horizontal perpendicular (rotate +90° in xy)
#   e3 = vertical (z)
print("\n[B.2] Velocity-aligned body-frame δ (along / cross_h / vertical)")
v_last = (X_train[:, -1] - X_train[:, -2]) / DT
v_last_xy = v_last[:, :2]
speed_xy = np.linalg.norm(v_last_xy, axis=-1, keepdims=True)
# guard tiny
e1_xy = np.where(speed_xy > 1e-6, v_last_xy / np.maximum(speed_xy, 1e-12), np.array([[1.0,0.0]]))
# e1 in 3D (vz=0), e2 = perpendicular horizontal
e1 = np.concatenate([e1_xy, np.zeros((10000,1))], axis=-1)             # (N,3)
e2 = np.concatenate([-e1_xy[:,1:2], e1_xy[:,0:1], np.zeros((10000,1))], axis=-1)
e3 = np.tile(np.array([0,0,1.0]), (10000,1))

# Same for test
v_last_te = (X_test[:, -1] - X_test[:, -2]) / DT
v_last_xy_te = v_last_te[:, :2]
speed_xy_te = np.linalg.norm(v_last_xy_te, axis=-1, keepdims=True)
e1_xy_te = np.where(speed_xy_te > 1e-6, v_last_xy_te / np.maximum(speed_xy_te, 1e-12), np.array([[1.0,0.0]]))
e1_te = np.concatenate([e1_xy_te, np.zeros((10000,1))], axis=-1)
e2_te = np.concatenate([-e1_xy_te[:,1:2], e1_xy_te[:,0:1], np.zeros((10000,1))], axis=-1)
e3_te = np.tile(np.array([0,0,1.0]), (10000,1))

def hit_body_offset(pred, y, e1, e2, e3, da, db, dc):
    # delta in body frame: da*e1 + db*e2 + dc*e3
    delta = da*e1 + db*e2 + dc*e3
    d = np.linalg.norm(pred + delta - y, axis=-1)
    return (d <= 0.01).mean()

# coarse body grid
best_b = (base_hit, np.array([0.0,0.0,0.0]))
for da in coarse_range:
    for db in coarse_range:
        for dc in coarse_range:
            h = hit_body_offset(oof, y_train, e1, e2, e3, da, db, dc)
            if h > best_b[0]: best_b = (h, np.array([da,db,dc]))
print(f"  coarse body best: (along, cross_h, vert) = ({best_b[1][0]*1000:+.2f}, {best_b[1][1]*1000:+.2f}, {best_b[1][2]*1000:+.2f}) mm  hit = {best_b[0]:.4f}  Δ = {best_b[0]-base_hit:+.4f}")

c = best_b[1]; best_b_fine = best_b
for da in c[0] + fine_range:
    for db in c[1] + fine_range:
        for dc in c[2] + fine_range:
            h = hit_body_offset(oof, y_train, e1, e2, e3, da, db, dc)
            if h > best_b_fine[0]: best_b_fine = (h, np.array([da,db,dc]))
hit_body, delta_body = best_b_fine
print(f"  fine body best:   (along, cross_h, vert) = ({delta_body[0]*1000:+.3f}, {delta_body[1]*1000:+.3f}, {delta_body[2]*1000:+.3f}) mm  hit = {hit_body:.4f}  Δ = {hit_body-base_hit:+.4f}")

# ============================================================
# (3) Speed-conditional global δ (over-fit 우려 → 분석만)
# ============================================================
print("\n[B.3] Speed-conditional global δ (5 bins by |v_last|)")
sp_last = np.linalg.norm(v_last, axis=-1)
bins = np.quantile(sp_last, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
bin_idx = np.clip(np.searchsorted(bins[1:-1], sp_last), 0, 4)
print(f"  speed quantile bins: {[f'{b:.2f}' for b in bins]}")
deltas_per_bin = np.zeros((5,3))
hit_per_bin = np.zeros(5); hit_base_per_bin = np.zeros(5)
for b in range(5):
    m = bin_idx == b
    o_b = oof[m]; y_b = y_train[m]
    hit_base_per_bin[b] = (np.linalg.norm(o_b - y_b, axis=-1) <= 0.01).mean()
    # fine grid only
    best_bin = (hit_base_per_bin[b], np.array([0.0,0.0,0.0]))
    for dx in np.arange(-2, 2.01, 0.25) * 1e-3:
        for dy in np.arange(-2, 2.01, 0.25) * 1e-3:
            for dz in np.arange(-2, 2.01, 0.25) * 1e-3:
                d = np.linalg.norm(o_b + np.array([dx,dy,dz]) - y_b, axis=-1)
                h = (d<=0.01).mean()
                if h > best_bin[0]: best_bin = (h, np.array([dx,dy,dz]))
    deltas_per_bin[b] = best_bin[1]; hit_per_bin[b] = best_bin[0]
    print(f"  bin {b} (n={m.sum():>4d}, speed [{bins[b]:.2f},{bins[b+1]:.2f}]): δ=({best_bin[1][0]*1000:+.2f},{best_bin[1][1]*1000:+.2f},{best_bin[1][2]*1000:+.2f})mm  hit {hit_base_per_bin[b]:.4f} → {best_bin[0]:.4f}  Δ {best_bin[0]-hit_base_per_bin[b]:+.4f}")

# apply speed-conditional → compute aggregate (potentially overfit)
oof_sp = oof.copy()
for b in range(5):
    m = bin_idx == b
    oof_sp[m] += deltas_per_bin[b]
hit_sp = (np.linalg.norm(oof_sp - y_train, axis=-1) <= 0.01).mean()
print(f"  speed-conditional aggregate hit = {hit_sp:.4f}  Δ = {hit_sp - base_hit:+.4f}  (오버핏 가능성)")

# ============================================================
# Gate & decision
# ============================================================
GATE = 0.0008
candidates = {
    "global3D": (hit_global, delta_global, "x/y/z 3D 오프셋"),
    "bodyAxis": (hit_body, delta_body, "along/cross_h/vert body-frame 오프셋"),
}
print("\n[gate] +0.0008 threshold")
for k, (h, d, desc) in candidates.items():
    passed = (h - base_hit) >= GATE
    print(f"  {k:<10}: lift {h-base_hit:+.4f}  {'PASS' if passed else 'FAIL'}")

# Best gating candidate
best_name = max(candidates.keys(), key=lambda k: candidates[k][0])
best_hit_v, best_delta, best_desc = candidates[best_name]
best_lift = best_hit_v - base_hit
print(f"\n[best] {best_name}: lift {best_lift:+.4f}  ({best_desc})")

if best_lift >= GATE:
    # Apply to test
    if best_name == "global3D":
        test_offset = test_pred + best_delta
    else:
        # bodyAxis: apply per-sample using test body frame
        test_offset = test_pred + best_delta[0]*e1_te + best_delta[1]*e2_te + best_delta[2]*e3_te
    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    sub_ids = sample["id"].values
    test_df = pd.DataFrame({"id": sub_ids, "x": test_offset[:,0], "y": test_offset[:,1], "z": test_offset[:,2]})
    out_path = FINAL / f"submission_v112_offset_{best_name}_oof{best_hit_v:.4f}.csv"
    test_df.to_csv(out_path, index=False)
    print(f"\n[write] {out_path}")
else:
    print("\n[skip] gate FAIL — submission 생성 안 함, v112 유지")

# ============================================================
# Report
# ============================================================
md = []
md.append("# STEP B — Hit-Aware Offset 결과")
md.append("")
md.append("날짜: 2026-05-25 (STEP A 후 cheap offset 탐색)")
md.append("")
md.append(f"기준: v112_v107_diverse OOF, base hit1cm = **{base_hit:.4f}**")
md.append("")
md.append("## B.1 Global 3D δ 그리드 (±3mm coarse 0.5mm → ±0.5mm fine 0.1mm)")
md.append("")
md.append(f"- best δ = ({delta_global[0]*1000:+.3f}, {delta_global[1]*1000:+.3f}, {delta_global[2]*1000:+.3f}) mm")
md.append(f"- hit = {hit_global:.4f}  (Δ = {hit_global-base_hit:+.4f})")
md.append(f"- residual mean (bias) = ({resid.mean(axis=0)[0]*1000:+.3f}, {resid.mean(axis=0)[1]*1000:+.3f}, {resid.mean(axis=0)[2]*1000:+.3f}) mm")
md.append("")
md.append("## B.2 Body-frame δ (along / cross_h / vertical)")
md.append("")
md.append(f"- best δ = ({delta_body[0]*1000:+.3f} along, {delta_body[1]*1000:+.3f} cross_h, {delta_body[2]*1000:+.3f} vert) mm")
md.append(f"- hit = {hit_body:.4f}  (Δ = {hit_body-base_hit:+.4f})")
md.append("")
md.append("## B.3 Speed-conditional global δ (5 bins, fine 0.25mm 그리드, 오버핏 우려 — 분석만)")
md.append("")
md.append("| bin | speed range (m/s) | n | base hit | δ_x | δ_y | δ_z | new hit | Δ |")
md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for b in range(5):
    md.append(f"| {b} | [{bins[b]:.2f},{bins[b+1]:.2f}] | {int((bin_idx==b).sum())} | {hit_base_per_bin[b]:.4f} | {deltas_per_bin[b,0]*1000:+.2f} | {deltas_per_bin[b,1]*1000:+.2f} | {deltas_per_bin[b,2]*1000:+.2f} | {hit_per_bin[b]:.4f} | {hit_per_bin[b]-hit_base_per_bin[b]:+.4f} |")
md.append(f"")
md.append(f"- aggregate speed-conditional hit = {hit_sp:.4f}  (Δ = {hit_sp-base_hit:+.4f})  ← bin별 fit, train OOF에 ad-hoc")
md.append("")
md.append("## 결정")
md.append("")
md.append(f"- gate: +{GATE:.4f}")
md.append(f"- best gating candidate: **{best_name}** lift = {best_lift:+.4f}")
if best_lift >= GATE:
    md.append(f"- → **PASS**. submission 생성: `final_candidates/submission_v112_offset_{best_name}_oof{best_hit_v:.4f}.csv`")
else:
    md.append(f"- → **FAIL** (gate {GATE} 미달). v112 유지.")
md.append("")

(REPORTS / "hit_offset.md").write_text("\n".join(md), encoding="utf-8")
print(f"\n[done] reports/hit_offset.md 저장")

# JSON
(REPORTS / "hit_offset.json").write_text(json.dumps({
    "base_hit": float(base_hit),
    "global3D": {"delta_mm": (delta_global*1000).tolist(), "hit": float(hit_global), "lift": float(hit_global-base_hit)},
    "bodyAxis": {"delta_mm": (delta_body*1000).tolist(), "hit": float(hit_body), "lift": float(hit_body-base_hit)},
    "speed_conditional_aggregate_hit": float(hit_sp),
    "gate": GATE,
    "best_name": best_name,
    "best_lift": float(best_lift),
    "passed": bool(best_lift >= GATE),
}, indent=2), encoding="utf-8")
