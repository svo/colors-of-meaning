# Rate-distortion frontier for semantic color compression

The ~1024:1 headline is one operating point; this report measures the whole frontier.
Each codec is swept across bit budgets and its native distortion recorded: color-VQ over
grid resolutions (bits = log2(bins)), Product Quantization over subquantizers matched to the
same bits, and gzip as a single data-dependent point. The color codec additionally records a
downstream retrieval accuracy at each budget, so the cost of compression is shown in both
perceptual distortion (ΔE for color-VQ, MSE for gzip/PQ) and task accuracy at matched budgets.

Library versions: numpy 2.4.6, scikit-learn 1.9.0.

## Rate-distortion points

| method | bits/token | distortion | metric | accuracy |
|---|---|---|---|---|
| color_vq | 3.00 | 153.674758 | ΔE | 0.1600 |
| color_vq | 6.00 | 41.806017 | ΔE | 0.2567 |
| color_vq | 9.00 | 15.044891 | ΔE | 0.1767 |
| color_vq | 12.00 | 6.855841 | ΔE | 0.1700 |
| gzip | 11393.39 | 0.000000 | MSE | n/a |
| pq | 3.00 | 0.001845 | MSE | n/a |
| pq | 6.00 | 0.001889 | MSE | n/a |
| pq | 9.00 | 0.001891 | MSE | n/a |
| pq | 12.00 | 0.001862 | MSE | n/a |

## Matched-budget comparison

| bits/token | method | distortion | metric |
|---|---|---|---|
| 3.00 | color_vq | 153.674758 | ΔE |
| 3.00 | pq | 0.001845 | MSE |
| 6.00 | color_vq | 41.806017 | ΔE |
| 6.00 | pq | 0.001889 | MSE |
| 9.00 | color_vq | 15.044891 | ΔE |
| 9.00 | pq | 0.001891 | MSE |
| 12.00 | color_vq | 6.855841 | ΔE |
| 12.00 | pq | 0.001862 | MSE |

## Pareto frontier

The envelope is the geometric lower-left set over (bits, native distortion). Distortion
metrics differ across codecs (ΔE for color-VQ, MSE for gzip/PQ), so cross-codec domination
is not directly comparable; read each codec's own curve in the figure rather than comparing
ΔE against MSE as if they were one number.

| method | bits/token | distortion | metric |
|---|---|---|---|
| color_vq | 3.00 | 153.674758 | ΔE |
| pq | 3.00 | 0.001845 | MSE |
| color_vq | 6.00 | 41.806017 | ΔE |
| color_vq | 9.00 | 15.044891 | ΔE |
| color_vq | 12.00 | 6.855841 | ΔE |
| gzip | 11393.39 | 0.000000 | MSE |

## Reproduce

```bash
tox -e rate_distortion -- --source documents --documents-dir documents --split-strategy work --min-paragraph-chars 200 --paragraphs-per-work 60 --validation-fraction 0.2 --test-fraction 0.2 --budgets 2 4 8 16 --methods color_vq gzip pq --with-accuracy --distance jensen_shannon --max-samples 300 --config configs/documents.yaml
```
