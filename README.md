# BioHub Cell Tracking

A standalone Kaggle pipeline for detecting and tracking cells in anisotropic
3D microscopy time series. The notebook reads the competition test volumes at
runtime, so it works with both the visible examples and Kaggle's hidden test
set.

## Current status

- Kaggle notebook: sarthaksharma14/biohub-standalone-physical-tracker
- Internet access: disabled
- External notebook outputs: none
- External model weights: none
- Competition submission: gated on output validation and explicit approval

## Method

1. Read each Zarr v3 timepoint directly from mounted competition data.
2. Preserve full Z resolution and downsample X/Y by four to make physical
   voxel spacing approximately isotropic.
3. Normalize each frame with robust intensity percentiles.
4. Detect cell centres with a two-scale Difference-of-Gaussians response.
5. Refine every centre using a local intensity-weighted centroid.
6. Match adjacent frames with Hungarian assignment in physical micrometres.
7. Reject links longer than 8.4 micrometres.
8. Remove isolated detections and serialize each dataset as a node block
   followed by its edge block.

The baseline intentionally does not invent divisions or temporal gaps. Those
events are rare in the training annotations, and false division edges have a
large precision cost.

## Submission guarantees

Before a dataset is written, the notebook verifies:

- positive contiguous node IDs;
- integer T/Z/Y/X coordinates inside the actual image shape;
- unique node IDs and edges;
- no dangling endpoints;
- links only between consecutive frames;
- at most one parent and one child per node;
- exact competition columns and dataset-block ordering;
- a consecutive global id column.

The CSV is streamed one dataset at a time to keep memory usage bounded when
Kaggle swaps in the much larger hidden test set.

## Visible validation

The completed Kaggle notebook produced 251,289 rows across all four visible
test movies in 81 seconds. Full artifact validation found:

- 129,502 node rows and 121,787 edge rows;
- zero nulls, duplicate IDs, dangling edges, or invalid coordinates;
- exact sample-compatible dataset and node/edge block ordering;
- visible adjusted edge Jaccard proxy: 0.7331;
- visible mean sparse-node recall: 0.8774.

The proxy uses the visible movies that also have matching training graphs. It
is not a private leaderboard score and is used only to compare notebook
variants before submission.

## Project files

- adaptive-graph/biohub_adaptive_graph_final.py: standalone Kaggle inference
  and submission pipeline
- adaptive-graph/kernel-metadata.json: private Kaggle notebook configuration
- biohub_baseline.py: original lightweight baseline
- detection-diagnostic/: detection experiments
- metric-validation/: metric validation experiments

## Run on Kaggle

    kaggle kernels push -p .\adaptive-graph
    kaggle kernels status sarthaksharma14/biohub-standalone-physical-tracker

After completion, download and validate the artifacts before creating a
competition submission.
