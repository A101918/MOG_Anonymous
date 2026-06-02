# MOG Anonymous — fused & final scores

Per-dataset directories with frame-level anomaly scores for MSAD, ShanghaiTech, UCF_Crime, and XD_Violence.

## Layout

Each dataset folder contains:

- `gt_concat.npy` — concatenated frame-level ground truth
- `fused_scores/` — per-video F2 fusion scores (pre-smoothing)
- `fused_scores_smoothed/` — per-video F2 fusion scores with Gaussian smoothing

## Evaluate micro frame-level AUROC

Requires Python 3 with `numpy` and `scikit-learn`.

```bash
python3 eval_score_verification_concat_aucroc.py
```

Optional flags:

```bash
python3 eval_score_verification_concat_aucroc.py --root .
python3 eval_score_verification_concat_aucroc.py --datasets MSAD UCF_Crime
python3 eval_score_verification_concat_aucroc.py --strict-length
```

The script concatenates per-video `.npy` files (sorted by filename), compares them to `gt_concat.npy`, and writes:

- `aucroc_concat_eval.json`
- `aucroc_concat_eval.csv`

Results are also printed as a table on stdout.
