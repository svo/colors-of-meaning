# Scaled, multi-dataset evaluation of the color method

Committed evidence for the color method beyond the 400-sample AG News budget. Every row is produced
by the command below; the sliced-Wasserstein proxy is only trusted once it clears the fidelity gate.

Library versions: numpy 2.4.6, scipy 1.17.1, POT 0.9.6.post1.

## Distance proxy fidelity gate

| proxy | exact | spearman | accuracy_delta (pts) | pairs | threshold | max_delta | faithful |
|-------|-------|----------|----------------------|-------|-----------|-----------|----------|
| sliced_wasserstein | wasserstein | 0.9916 | 0.0000 | 1200 | 0.95 | 1.0 | yes |

## Results

| dataset | method | distance | budget | accuracy | macro_f1 | mrr | bits/token | seconds |
|---------|--------|----------|--------|----------|----------|-----|------------|---------|
| ag_news | color | sliced | 4000 | 0.8175 | 0.8178 | 0.0000 | 12.00 | 702.7 |
| imdb | color | sliced | 600 | 0.5483 | 0.5478 | 0.0000 | 12.00 | 222.7 |
| newsgroups | color | sliced | 600 | 0.1650 | 0.1535 | 0.0000 | 12.00 | 203.3 |

## Reproduce

```bash
tox -e eval_suite -- --datasets ag_news imdb newsgroups --distance sliced --budgets 4000 600 600 --fidelity-accuracy-delta 0.0 --config configs/agnews_full.yaml --mapper-type unconstrained
```
