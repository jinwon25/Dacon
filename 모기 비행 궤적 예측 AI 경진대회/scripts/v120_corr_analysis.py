"""v120_corr_analysis.py — residual correlation of v120 vs pool members.

LB 0.6888 plateau의 핵심 원인은 pool 멤버간 residual corr ~0.99.
v120 Neural ODE는 kalman residual 미사용 → corr<0.93 도달 여부 검증.

게이트:
  - residual corr (vs v94/v97/v107) 평균 < 0.93 = paradigm 진정성 확인
  - 단독 OOF ≥ 0.665 = pool 멤버 자격
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
from v23_train import load_data


def residual_corr(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray, mask=None):
    """3D residual correlation between two predictors."""
    if mask is not None:
        pred_a = pred_a[mask]; pred_b = pred_b[mask]; y = y[mask]
    ra = pred_a - y
    rb = pred_b - y
    # per-axis corr
    corrs = [float(np.corrcoef(ra[:, k], rb[:, k])[0, 1]) for k in range(3)]
    # 3D flattened corr
    corr_3d = float(np.corrcoef(ra.reshape(-1), rb.reshape(-1))[0, 1])
    return corrs, corr_3d


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "full"
    state_v120 = PROJECT_DIR / "cache" / f"v120_{tag}_state.npz"
    if not state_v120.exists():
        print(f"[FAIL] {state_v120} not found")
        return

    X_train, X_test, y_train, _ = load_data()
    N = len(y_train)
    print(f"N={N}, y_train shape={y_train.shape}")

    v120 = np.load(state_v120, allow_pickle=True)
    oof_v120 = v120["oof_global"]
    fold_mask = v120["fold_mask"]
    rh_v120 = float(v120["rh_oof"])
    print(f"\n[v120/{tag}] OOF R-Hit = {rh_v120:.4f}  covered={fold_mask.sum()}")

    pool_files = {
        "v94": PROJECT_DIR / "cache" / "v94_state.npz",
        "v97": PROJECT_DIR / "cache" / "v97_state.npz",
        "v97_cap1p5": PROJECT_DIR / "cache" / "v97_cap1p5_state.npz",
        "v104b": PROJECT_DIR / "cache" / "v104b_state.npz",
        "v107": PROJECT_DIR / "cache" / "v107_state.npz",
        "v108_15_15_08": PROJECT_DIR / "cache" / "v108_15_15_08_state.npz",
    }
    rows = []
    for name, path in pool_files.items():
        if not path.exists(): continue
        d = np.load(path, allow_pickle=True)
        # find OOF field
        key = None
        for k in ("oof_v91", "oof"):
            if k in d.files: key = k; break
        if key is None:
            print(f"[skip] {name}: no OOF key, available={d.files}"); continue
        pool_oof = d[key]
        if pool_oof.shape != y_train.shape:
            print(f"[skip] {name}: shape {pool_oof.shape} != {y_train.shape}"); continue
        # rhit on fold_mask
        rh_pool = float((np.linalg.norm(pool_oof[fold_mask] - y_train[fold_mask], axis=-1) <= 0.01).mean())
        corr_axes, corr_3d = residual_corr(oof_v120, pool_oof, y_train, mask=fold_mask)
        rows.append({
            "model": name, "rh": rh_pool,
            "corr_x": corr_axes[0], "corr_y": corr_axes[1], "corr_z": corr_axes[2],
            "corr_3d": corr_3d,
        })

    print(f"\n{'model':<20} {'RH':>6}  {'corr_x':>7} {'corr_y':>7} {'corr_z':>7}  {'corr_3d':>7}  gate")
    for r in rows:
        gate_ok = "OK" if r["corr_3d"] < 0.93 else "FAIL"
        print(f"{r['model']:<20} {r['rh']:.4f}  "
              f"{r['corr_x']:.4f}  {r['corr_y']:.4f}  {r['corr_z']:.4f}  "
              f"{r['corr_3d']:.4f}  {gate_ok}")

    mean_corr = np.mean([r["corr_3d"] for r in rows]) if rows else float("nan")
    print(f"\nmean corr_3d = {mean_corr:.4f}  (gate <0.93)")

    # save
    out_md = PROJECT_DIR / "reports" / f"v120_{tag}_corr.md"
    out_md.parent.mkdir(exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# v120 residual correlation ({tag})\n\n")
        f.write(f"- v120 OOF R-Hit (covered {fold_mask.sum()}/{N}): **{rh_v120:.4f}**\n")
        f.write(f"- mean corr_3d vs pool = **{mean_corr:.4f}** (gate <0.93)\n\n")
        f.write(f"| model | RH | corr_x | corr_y | corr_z | corr_3d | gate |\n")
        f.write(f"|---|---:|---:|---:|---:|---:|:---|\n")
        for r in rows:
            gate_ok = "OK" if r["corr_3d"] < 0.93 else "FAIL"
            f.write(f"| {r['model']} | {r['rh']:.4f} | {r['corr_x']:.4f} | "
                    f"{r['corr_y']:.4f} | {r['corr_z']:.4f} | "
                    f"**{r['corr_3d']:.4f}** | {gate_ok} |\n")
    print(f"\n[saved] {out_md}")


if __name__ == "__main__":
    main()
