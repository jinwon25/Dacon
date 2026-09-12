# Public directional FiCR factor probes — 2026-07-27

## Outcome

The frozen public incumbent remains submission `1502437` at `0.6461250914`.
External ECMWF, DWD ICON, and CMC GEM direct/pooled forecasts were rejected:
Q2-selected policies reversed in Q3 or Q4, so no external-model prediction was
promoted.

A new calibration uses only directions already confirmed by public submissions:

- group 1: the positive direction of the pooled KMA/JMA expert;
- group 2: both directions of the pooled KMA/JMA expert;
- group 3: a small positive settlement offset on the KMA incumbent.

The historical 2024 macro delta is:

| component | delta |
|---|---:|
| score | `+0.0043876557` |
| 1-NMAE | `-0.0009826741` |
| FiCR | `+0.0097579855` |

All Q1, Q2, and H2 score deltas are positive. All 12 monthly macro score deltas
are positive; the worst is `+0.0010563977`. Complete issue-cycle bootstrap gives
a score q05 of `+0.0018675640` and a 99.77% positive-score fraction. The trade is
intentional: the candidate sacrifices 1-NMAE for threshold settlement revenue.

Adding the historical delta one-for-one to the public incumbent gives only a
scenario, not a forecast:

- score `0.6505127471`
- 1-NMAE `0.8748015736`
- FiCR `0.4262239207`

The parameters were screened on the repeatedly inspected 2024 OOF benchmark, so
the candidate is a manual public probe and is not private-safe evidence.

## Exact two-probe protocol

The official metric is a macro average over three independently scored groups.
The target was therefore split into complementary files:

1. `public_directional_ficr065_g1_probe_20260727.csv` changes G1 only.
2. `public_directional_ficr065_g2g3_probe_20260727.csv` changes G2 and G3 only.
3. `public_directional_ficr065_target_20260727.csv` contains both changes.

After the first two files are scored, every target metric is known exactly:

```text
target = G1 probe + G2/G3 probe - incumbent
```

The target should use the third daily slot only when the exact computed score
meets the chosen threshold. This avoids spending a submission on a composition
whose score is already algebraically known.

## Safety control

`public_directional_safe_g2g3_20260727.csv` is the component-safe control. Its
historical macro delta is `+0.0012696217` score, `+0.0001617364` 1-NMAE, and
`+0.0023775070` FiCR. All issue-block bootstrap draws improve every component,
but one month is negative and its one-for-one score scenario remains below
`0.65`; it is not first in the public submission order.

## Governance

- No 2025 actual generation or target-time observation is used.
- Public scores identify already-positive signal families, not test labels.
- The new row gates and step sizes use contaminated historical OOF and are
  labeled accordingly.
- All four CSVs pass schema, identifier, row-count, finite-value, and capacity
  range validation.
- No submission was made automatically.

## Public results

The two complementary probes were submitted manually:

| submission | changed groups | score | 1-NMAE | FiCR |
|---:|---|---:|---:|---:|
| `1503281` | G1 | `0.6441230595` | `0.8738627757` | `0.4143833434` |
| `1503282` | G2/G3 | `0.6451061773` | `0.8752080619` | `0.4150042926` |

Both probes lose score, 1-NMAE, and FiCR versus incumbent `1502437`. Their
macro deltas are:

- G1 probe: score `-0.0020020319`, 1-NMAE `-0.0019214720`, FiCR
  `-0.0020825918`.
- G2/G3 probe: score `-0.0010189141`, 1-NMAE `-0.0005761858`, FiCR
  `-0.0014616426`.

The target composition is therefore known without submission:

```text
score  = 0.6441230595 + 0.6451061773 - 0.6461250914 = 0.6431041454
1-NMAE = 0.8738627757 + 0.8752080619 - 0.8757842477 = 0.8732865899
FiCR   = 0.4143833434 + 0.4150042926 - 0.4164659352 = 0.4129217008
```

Decision: reject the target, close the aggressive G1 calibration, close the
aggressive G2/G3 calibration, and withhold the correlated safe control. The
final daily slot is preserved rather than spent on a known-negative target.
