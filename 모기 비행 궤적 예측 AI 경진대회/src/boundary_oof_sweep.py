from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import pipeline as p


@dataclass(frozen=True)
class BoundaryConfig:
    cap: float
    apply_scale: float
    seed: int

    @property
    def name(self) -> str:
        return (
            f"cap{_fmt_float(self.cap)}"
            f"_apply{_fmt_float(self.apply_scale)}"
            f"_seed{self.seed}"
        )


def _fmt_float(value: float) -> str:
    text = f"{value:g}".replace("-", "m").replace(".", "p")
    return text


def _parse_config(text: str) -> BoundaryConfig:
    parts = [x.strip() for x in re.split(r"[:,/]", text) if x.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"config must be cap:apply_scale:seed, got {text!r}"
        )
    return BoundaryConfig(float(parts[0]), float(parts[1]), int(parts[2]))


def _run_one_fold(
    *,
    cfg: BoundaryConfig,
    fold: int,
    folds: int,
    selector_out: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Path:
    fold_dir = out_dir / cfg.name / f"fold{fold}"
    report_path = fold_dir / "boundary_tiny_correction_report.json"
    val_path = fold_dir / "boundary_val_predictions.npz"
    if args.reuse and report_path.exists() and val_path.exists():
        return fold_dir

    argv = [
        "--root",
        p.DATA_ROOT,
        "--out-dir",
        fold_dir,
        "--fold",
        fold,
        "--folds",
        folds,
        "--score-bank",
        selector_out / "oof_selector_scores.npz",
        "--epochs",
        args.epochs,
        "--fine-epochs",
        args.fine_epochs,
        "--min-epochs",
        args.min_epochs,
        "--patience",
        args.patience,
        "--hidden",
        args.hidden,
        "--batch",
        args.batch,
        "--lr",
        args.lr,
        "--fine-lr-scale",
        args.fine_lr_scale,
        "--cap",
        cfg.cap,
        "--apply-scale",
        cfg.apply_scale,
        "--device",
        args.device,
        "--seed",
        cfg.seed,
        "--save-val-pred",
    ]
    p.call_main(p.BOUNDARY_MAIN, argv)
    return fold_dir


def _collect_config(
    *,
    cfg: BoundaryConfig,
    cfg_dir: Path,
    folds: int,
) -> dict[str, object]:
    ids, train_y = p.read_labels(p.DATA_ROOT / "train_labels.csv")
    n = len(train_y)
    oof: dict[str, np.ndarray] = {
        "soft": np.full((n, 3), np.nan, dtype=np.float32),
        "gate": np.full((n, 3), np.nan, dtype=np.float32),
        "argmax": np.full((n, 3), np.nan, dtype=np.float32),
    }
    fold_reports: list[dict[str, object]] = []
    covered = np.zeros(n, dtype=bool)

    for fold in range(folds):
        fold_dir = cfg_dir / f"fold{fold}"
        report = json.loads(
            (fold_dir / "boundary_tiny_correction_report.json").read_text(
                encoding="utf-8"
            )
        )
        val = np.load(fold_dir / "boundary_val_predictions.npz", allow_pickle=True)
        mask = val["val_mask"].astype(bool)
        covered |= mask
        for mode in oof:
            oof[mode][mask] = val[mode].astype(np.float32)
        fold_reports.append(
            {
                "fold": fold,
                "soft": report["soft"]["metrics"],
                "gate": report["gate"]["metrics"],
                "argmax": report["argmax"],
                "gate_temperature": report["gate"].get("temperature"),
                "gate_margin_quantile": report["gate"].get("margin_quantile"),
                "gate_argmax_rate": report["gate"].get("argmax_rate"),
            }
        )

    if not np.all(covered):
        missing = int(np.sum(~covered))
        raise RuntimeError(f"{cfg.name}: missing OOF rows: {missing}")

    summary: dict[str, object] = {
        "config": {
            "name": cfg.name,
            "cap": cfg.cap,
            "apply_scale": cfg.apply_scale,
            "seed": cfg.seed,
        },
        "folds": folds,
        "covered_rows": int(np.sum(covered)),
        "fold_reports": fold_reports,
        "oof": {
            mode: p.metrics(pred, train_y)
            for mode, pred in oof.items()
        },
    }
    np.savez_compressed(
        cfg_dir / "boundary_oof_predictions.npz",
        ids=np.asarray(ids, dtype=object),
        y=train_y.astype(np.float32),
        **{f"{mode}_pred": pred for mode, pred in oof.items()},
    )
    (cfg_dir / "boundary_oof_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def _write_index(out_dir: Path, summaries: list[dict[str, object]]) -> None:
    ranked = sorted(
        summaries,
        key=lambda row: (
            row["oof"]["gate"]["hit"],  # type: ignore[index]
            -row["oof"]["gate"]["mean"],  # type: ignore[index]
        ),
        reverse=True,
    )
    lines = [
        "# Boundary OOF Sweep",
        "",
        "Ranked by 5-fold OOF gate hit, then lower mean distance.",
        "",
        "| rank | config | gate hit | gate hits | gate mean | gate q95 | soft hit | argmax hit |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, start=1):
        oof = row["oof"]  # type: ignore[assignment]
        gate = oof["gate"]  # type: ignore[index]
        soft = oof["soft"]  # type: ignore[index]
        argmax = oof["argmax"]  # type: ignore[index]
        cfg = row["config"]  # type: ignore[assignment]
        lines.append(
            "| "
            f"{rank} | {cfg['name']} | "
            f"{gate['hit']:.6f} | {gate['hits']} | "
            f"{gate['mean']:.8f} | {gate['q95']:.8f} | "
            f"{soft['hit']:.6f} | {argmax['hit']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Use this table to choose boundary settings before spending a public LB submission.",
            "Public LB should be treated as a final sanity check, not the tuning loop.",
            "",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run boundary tiny-correction across all folds and aggregate OOF metrics."
    )
    parser.add_argument(
        "--config",
        action="append",
        type=_parse_config,
        default=[],
        help="Boundary config as cap:apply_scale:seed. Can be repeated.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--selector-out", type=Path, default=p.WORK_DIR / "selector_full")
    parser.add_argument("--out-dir", type=Path, default=p.WORK_DIR / "02_boundary_oof")
    parser.add_argument("--reuse", action="store_true", help="Reuse completed fold folders.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--fine-epochs", type=int, default=1)
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--fine-lr-scale", type=float, default=0.18)
    args = parser.parse_args()

    configs = args.config or [
        BoundaryConfig(0.006, 1.0, 20260606),
        BoundaryConfig(0.006, 0.75, 20260606),
    ]
    selector_out = args.selector_out.resolve()
    if not (selector_out / "oof_selector_scores.npz").exists():
        raise FileNotFoundError(selector_out / "oof_selector_scores.npz")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for cfg in configs:
        cfg_dir = out_dir / cfg.name
        cfg_dir.mkdir(parents=True, exist_ok=True)
        for fold in range(args.folds):
            print(f"[OOF] config={cfg.name} fold={fold}/{args.folds - 1}", flush=True)
            _run_one_fold(
                cfg=cfg,
                fold=fold,
                folds=args.folds,
                selector_out=selector_out,
                out_dir=out_dir,
                args=args,
            )
        summary = _collect_config(cfg=cfg, cfg_dir=cfg_dir, folds=args.folds)
        summaries.append(summary)
        gate = summary["oof"]["gate"]  # type: ignore[index]
        print(
            "[OOF_SUMMARY]",
            cfg.name,
            f"gate_hit={gate['hit']:.6f}",
            f"hits={gate['hits']}",
            f"mean={gate['mean']:.8f}",
            flush=True,
        )

    (out_dir / "boundary_oof_sweep_summary.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    _write_index(out_dir, summaries)
    print(f"[DONE] wrote {out_dir / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
