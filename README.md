# MOG Anonymous — fused score verification

Paper Table-1 fusion setup (λ_base / λ_motion / λ_anchor = 0.3 / 0.4 / 0.7, θ_trigger = 0.4).

Per-dataset directories contain `fused_scores/` and `fused_scores_smoothed/` frame-level `.npy` files.

## Micro frame-level AUROC (concatenated)

| Dataset | fused_scores | fused_scores_smoothed |
|---------|-------------:|----------------------:|
| MSAD | 0.848394 | 0.878817 |
| ShanghaiTech | 0.758743 | 0.789671 |
| UCF_Crime | 0.779910 | 0.842624 |
| XD_Violence | 0.885959 | 0.919126 |
