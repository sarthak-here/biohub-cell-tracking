# BioHub Cell Tracking

A Kaggle notebook pipeline for tracking cells through anisotropic 3D microscopy time series. The current candidate uses a public learned tracker as its pretrained graph source, then applies dataset-specific physical graph calibration and strict topology validation.

## Current status

- Kaggle notebook: `sarthaksharma14/biohub-adaptive-graph-calibration`
- Notebook status: complete
- Competition submission: not submitted yet
- Data runs entirely through Kaggle-mounted competition data
- Local dataset download is not required

## Method

1. Load the output graph from the public learned BioHub tracker notebook.
2. Match predictions to visible training tracks within the official 7 micrometer node radius.
3. Evaluate candidate edge limits from 6 to 14 micrometers for each movie.
4. Keep only consecutive-frame links.
5. Enforce one incoming edge per node and at most two outgoing edges for cell division.
6. Write and structurally validate the final submission artifact.

## Validation

The selected 6 micrometer limit produced the following visible-label proxy results:

| Dataset | Node recall | Edge Jaccard |
| --- | ---: | ---: |
| 44b6_0113de3b | 1.000 | 0.980 |
| 44b6_0b24845f | 1.000 | 0.918 |
| 6bba_05b6850b | 0.997 | 0.985 |
| 6bba_05db0fb1 | 0.997 | 0.934 |
| Aggregate | 0.998 mean | 0.955 |

This is a local proxy calculated from visible labels, not a Kaggle leaderboard score.

Final artifact checks:

- 4 test datasets
- 136,809 nodes
- 130,263 edges
- 0 duplicate nodes
- 0 dangling or non-consecutive edges
- Maximum in-degree: 1
- Maximum out-degree: 2
- Maximum edge length: 5.998 micrometers

## Project files

- `adaptive-graph/biohub_adaptive_graph_final.py`: final Kaggle candidate
- `adaptive-graph/kernel-metadata.json`: private Kaggle notebook configuration
- `biohub_baseline.py`: lightweight local-maxima baseline
- `detection-diagnostic/`: bright/dark peak experiments
- `metric-validation/`: visible-label metric prototype

## Run on Kaggle

```powershell
kaggle kernels push -p .\adaptive-graph
kaggle kernels status sarthaksharma14/biohub-adaptive-graph-calibration
```

The final competition submission is intentionally gated on explicit approval after validation.
