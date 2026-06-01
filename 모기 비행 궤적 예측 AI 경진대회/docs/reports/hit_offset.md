# STEP B — Hit-Aware Offset 결과

날짜: 2026-05-25 (STEP A 후 cheap offset 탐색)

기준: v112_v107_diverse OOF, base hit1cm = **0.6768**

## B.1 Global 3D δ 그리드 (±3mm coarse 0.5mm → ±0.5mm fine 0.1mm)

- best δ = (+0.000, +0.000, +0.000) mm
- hit = 0.6768  (Δ = +0.0000)
- residual mean (bias) = (-0.021, +0.172, -0.104) mm

## B.2 Body-frame δ (along / cross_h / vertical)

- best δ = (-0.000 along, -0.100 cross_h, -0.100 vert) mm
- hit = 0.6770  (Δ = +0.0002)

## B.3 Speed-conditional global δ (5 bins, fine 0.25mm 그리드, 오버핏 우려 — 분석만)

| bin | speed range (m/s) | n | base hit | δ_x | δ_y | δ_z | new hit | Δ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | [0.02,0.32] | 2000 | 0.8665 | -0.25 | +0.00 | -0.50 | 0.8690 | +0.0025 |
| 1 | [0.32,0.50] | 2000 | 0.7640 | +0.75 | -0.50 | +0.50 | 0.7690 | +0.0050 |
| 2 | [0.50,0.67] | 2000 | 0.6805 | -0.25 | +0.50 | -0.25 | 0.6835 | +0.0030 |
| 3 | [0.67,0.93] | 2000 | 0.5830 | +0.00 | +0.00 | -0.50 | 0.5855 | +0.0025 |
| 4 | [0.93,1.35] | 2000 | 0.4900 | +0.00 | +0.00 | +0.00 | 0.4900 | +0.0000 |

- aggregate speed-conditional hit = 0.6794  (Δ = +0.0026)  ← bin별 fit, train OOF에 ad-hoc

## 결정

- gate: +0.0008
- best gating candidate: **bodyAxis** lift = +0.0002
- → **FAIL** (gate 0.0008 미달). v112 유지.
