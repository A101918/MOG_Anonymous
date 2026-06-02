#!/usr/bin/env python3
"""
Micro frame-level AUC-ROC for Score_Verification export trees.

Expects each dataset directory under --root to contain:
  gt_concat.npy
  <score_dir>/*.npy   (e.g. fused_scores, fused_scores_smoothed)

Per-video predictions are concatenated in sorted filename order and
compared against gt_concat.npy (same layout as export_F2_fused_scores.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR
DEFAULT_DATASETS = ("MSAD", "ShanghaiTech", "UCF_Crime", "XD_Violence")
DEFAULT_SCORE_DIRS = ("fused_scores", "fused_scores_smoothed")


def load_ravel(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float64).ravel()


def concat_predictions(pred_dir: Path) -> Tuple[np.ndarray, int]:
    files = sorted(pred_dir.glob("*.npy"))
    if not files:
        return np.array([], dtype=np.float64), 0
    parts = [load_ravel(p) for p in files]
    return np.concatenate(parts), len(files)


def micro_auc(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[Optional[float], str]:
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if y_true.shape[0] != y_score.shape[0]:
        return None, f"length mismatch pred={y_score.shape[0]} gt={y_true.shape[0]}"
    if np.unique(y_true).size < 2:
        return None, "only one class in ground truth"
    return float(roc_auc_score(y_true, y_score)), ""


def discover_score_dirs(ds_path: Path) -> List[str]:
    found: List[str] = []
    for p in sorted(ds_path.iterdir()):
        if not p.is_dir():
            continue
        if any(p.glob("*.npy")):
            found.append(p.name)
    return found


def eval_dataset(
    ds_path: Path,
    score_dirs: List[str],
    *,
    strict_length: bool,
) -> Dict[str, Any]:
    gt_path = ds_path / "gt_concat.npy"
    if not gt_path.is_file():
        raise FileNotFoundError(f"missing {gt_path}")

    y_true = load_ravel(gt_path)
    out: Dict[str, Any] = {
        "dataset": ds_path.name,
        "gt_frames": int(y_true.size),
        "gt_positives": int(np.sum(y_true > 0.5)),
        "scores": {},
    }

    for sub in score_dirs:
        pred_dir = ds_path / sub
        entry: Dict[str, Any] = {"score_dir": sub}
        if not pred_dir.is_dir():
            entry["error"] = f"missing directory {pred_dir}"
            out["scores"][sub] = entry
            continue

        y_score, n_videos = concat_predictions(pred_dir)
        entry["n_videos"] = n_videos
        entry["pred_frames"] = int(y_score.size)

        if y_score.size == 0:
            entry["error"] = "no per-video .npy files"
            out["scores"][sub] = entry
            continue

        auc, err = micro_auc(y_true, y_score)
        entry["aucroc"] = auc
        if err:
            entry["error"] = err
            if strict_length:
                raise RuntimeError(f"{ds_path.name}/{sub}: {err}")

        out["scores"][sub] = entry

    return out


def print_table(results: List[Dict[str, Any]]) -> None:
    print("=" * 72)
    print(f"{'Dataset':<16} {'Score dir':<24} {'AUROC':>10}  Notes")
    print("-" * 72)
    for ds in results:
        name = ds["dataset"]
        for sub, info in ds["scores"].items():
            auc = info.get("aucroc")
            auc_s = f"{auc:.6f}" if auc is not None else "N/A"
            notes = []
            if info.get("error"):
                notes.append(info["error"])
            if "n_videos" in info:
                notes.append(f"videos={info['n_videos']} frames={info.get('pred_frames')}")
            note = "; ".join(notes)
            print(f"{name:<16} {sub:<24} {auc_s:>10}  {note}")
    print("=" * 72)


def write_csv(path: Path, results: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "dataset",
                "score_dir",
                "aucroc",
                "n_videos",
                "pred_frames",
                "gt_frames",
                "error",
            ]
        )
        for ds in results:
            gt_frames = ds.get("gt_frames")
            for sub, info in ds["scores"].items():
                w.writerow(
                    [
                        ds["dataset"],
                        sub,
                        info.get("aucroc"),
                        info.get("n_videos"),
                        info.get("pred_frames"),
                        gt_frames,
                        info.get("error", ""),
                    ]
                )


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Concatenate per-video scores and compute micro frame AUC-ROC."
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Score_Verification root (default: {DEFAULT_ROOT})",
    )
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset subdirectories to evaluate",
    )
    ap.add_argument(
        "--score-dirs",
        nargs="+",
        default=None,
        help=(
            "Score subfolders under each dataset "
            f"(default: {', '.join(DEFAULT_SCORE_DIRS)}; "
            "use 'auto' to discover dirs containing .npy)"
        ),
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Write full results JSON (default: <root>/aucroc_concat_eval.json)",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Write CSV table (default: <root>/aucroc_concat_eval.csv)",
    )
    ap.add_argument(
        "--strict-length",
        action="store_true",
        help="Raise if pred length does not match gt_concat",
    )
    args = ap.parse_args(argv)

    root: Path = args.root.resolve()
    if not root.is_dir():
        print(f"error: --root is not a directory: {root}", file=sys.stderr)
        return 1

    out_json = args.out_json or (root / "aucroc_concat_eval.json")
    out_csv = args.out_csv or (root / "aucroc_concat_eval.csv")

    results: List[Dict[str, Any]] = []
    errors: List[str] = []

    for ds_name in args.datasets:
        ds_path = root / ds_name
        if not ds_path.is_dir():
            errors.append(f"missing dataset dir: {ds_path}")
            continue

        if args.score_dirs is None:
            score_dirs = list(DEFAULT_SCORE_DIRS)
        elif len(args.score_dirs) == 1 and args.score_dirs[0] == "auto":
            score_dirs = discover_score_dirs(ds_path)
        else:
            score_dirs = list(args.score_dirs)

        try:
            results.append(
                eval_dataset(ds_path, score_dirs, strict_length=args.strict_length)
            )
        except (FileNotFoundError, RuntimeError) as exc:
            errors.append(f"{ds_name}: {exc}")

    payload: Dict[str, Any] = {
        "root": str(root),
        "metric": "micro_frame_aucroc",
        "datasets": results,
        "errors": errors,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(out_csv, results)

    print_table(results)
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_csv}")

    if errors:
        for e in errors:
            print(f"warning: {e}", file=sys.stderr)
        return 1

    # Fail if any score dir has no auc
    for ds in results:
        for info in ds["scores"].values():
            if info.get("aucroc") is None:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
