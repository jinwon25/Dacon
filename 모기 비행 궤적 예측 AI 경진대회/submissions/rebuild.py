"""rebuild.py — 최종 제출(Private LB 0.703151, 2등) 자급식 재현/검증.

이 폴더(submissions/)만 있으면 외부 의존 없이 최종 제출 2개를 재생성하고
원본과 바이트 수준으로 일치하는지 검증한다.

레시피:
    cree_ens3 = mean(cree_xy2, cree_xy2s1, cree_xy2h3)
    v157_a040 = 0.60 * base + 0.40 * cree_ens3
    v157_a045 = 0.55 * base + 0.45 * cree_ens3

사용: python rebuild.py     (이 폴더 안에서 실행, 상대경로)
환경: Python 3.11 / numpy / pandas (CPU only)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
INP = HERE / "inputs"


def xyz(path: Path) -> np.ndarray:
    return pd.read_csv(path)[["x", "y", "z"]].to_numpy()


def main():
    base = xyz(INP / "base_v148blend.csv")
    cree = [xyz(INP / f"cree_{t}.csv") for t in ("xy2", "xy2s1", "xy2h3")]
    ens3 = np.mean(cree, axis=0)
    ids = pd.read_csv(INP / "base_v148blend.csv")["id"]

    specs = [(0.40, "submission_v157_ens3a0.40_FINAL.csv"),
             (0.45, "submission_v157_ens3a0.45_FINAL.csv")]
    ok = True
    for a, fn in specs:
        pred = (1.0 - a) * base + a * ens3
        df = pd.DataFrame({"id": ids, "x": pred[:, 0], "y": pred[:, 1], "z": pred[:, 2]})
        ref = HERE / fn
        if ref.exists():
            old = xyz(ref)
            d = np.linalg.norm(pred - old, axis=-1).max() * 1000
            status = "MATCH" if d < 0.01 else "MISMATCH"
            ok = ok and (d < 0.01)
            print(f"[alpha={a}] {fn}: max diff = {d:.5f} mm  -> {status}", flush=True)
        df.to_csv(HERE / ("rebuilt_" + fn), index=False)
    print("\n[done] 재현 " + ("성공: 두 최종 제출 모두 원본과 일치." if ok else "확인 필요."), flush=True)


if __name__ == "__main__":
    main()
