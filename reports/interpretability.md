# Interpretability validation of the structured color mapper

The structured mapper is *trained toward* hue=topic-cluster, lightness=sentiment, and
chroma=concreteness targets, so the claim is true by construction on the training split.
This report makes it falsifiable: each axis is measured on a held-out split for the
structured mapper and for a negative control (a mapper never trained toward these axes).
Interpretability is asserted only where the structured margin over the control clears the
documented threshold; any axis that fails is reported as a falsification, not a pass.

Method: each document's Lab color is read from its document embedding (the same granularity
the mapper is trained at); the sentiment signal is the IMDB binary label; concreteness is the
bundled Brysbaert lexicon score (offline). hue<->topic is normalized mutual information of
binned hue angle vs the gold class; the other two axes are rank/point-biserial correlation.

Library versions: numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.0.

Overall verdict: **VALIDATED**.
Falsified axes: none.

## Per-axis scores

| axis | structured | control | margin | threshold | verdict |
|------|-----------|---------|--------|-----------|---------|
| hue <-> topic (NMI) | 0.1293 | 0.0005 | +0.1288 | 0.0200 | pass |
| lightness <-> sentiment (corr) | 0.5718 | -0.0029 | +0.5747 | 0.0500 | pass |
| chroma <-> concreteness (corr) | 0.3312 | -0.0365 | +0.3677 | 0.0500 | pass |

## Reproduce

```bash
tox -e interpretability -- --dataset imdb --structured-model artifacts/models/structured_projector.pth --control noise --max-samples 400 --config configs/interpretability.yaml
```
