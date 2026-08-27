"""Offline physical graph calibration with competition-safe serialization."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCALE = np.array([1.625, 0.40625, 0.40625], np.float32)
EDGE_THRESHOLD_UM = 6.0
BOUNDS = {"t": (0, 99), "z": (0, 63), "y": (0, 255), "x": (0, 255)}


def find_source_submission():
    candidates = [
        path
        for path in Path("/kaggle/input").rglob("submission.csv")
        if "biohub-cell-tracking-learned-graph-w-gap-recovery" in str(path)
    ]
    if not candidates:
        raise FileNotFoundError("Required learned tracking graph was not mounted")
    return candidates[0]


def prepare_edge_geometry(nodes, edges):
    source = nodes.set_index("node_id")[["t", "z", "y", "x"]].add_prefix("source_")
    target = nodes.set_index("node_id")[["t", "z", "y", "x"]].add_prefix("target_")
    work = edges.join(source, on="source_id").join(target, on="target_id").dropna()
    work = work[work["target_t"].eq(work["source_t"] + 1)].copy()
    delta = (
        work[["target_z", "target_y", "target_x"]].to_numpy(np.float32)
        - work[["source_z", "source_y", "source_x"]].to_numpy(np.float32)
    ) * SCALE
    work["distance_um"] = np.sqrt((delta * delta).sum(axis=1))
    return work


def filter_edges(geometry):
    work = geometry[geometry["distance_um"].le(EDGE_THRESHOLD_UM)].sort_values("distance_um")
    work = work.drop_duplicates("target_id", keep="first")
    return work.groupby("source_id", sort=False, group_keys=False).head(2)


def normalize_submission(nodes, edges):
    nodes = nodes.copy()
    edges = edges.copy()
    clipped = 0
    for column, (minimum, maximum) in BOUNDS.items():
        before = nodes[column].copy()
        nodes[column] = nodes[column].clip(minimum, maximum)
        clipped += int(nodes[column].ne(before).sum())

    normalized_nodes = []
    normalized_edges = []
    for dataset in sorted(nodes["dataset"].unique()):
        movie_nodes = nodes[nodes["dataset"].eq(dataset)].copy()
        movie_edges = edges[edges["dataset"].eq(dataset)].copy()
        old_ids = movie_nodes["node_id"].astype(np.int64).tolist()
        mapping = {old_id: new_id for new_id, old_id in enumerate(old_ids, start=1)}
        movie_nodes["node_id"] = movie_nodes["node_id"].map(mapping)
        movie_edges["source_id"] = movie_edges["source_id"].map(mapping)
        movie_edges["target_id"] = movie_edges["target_id"].map(mapping)
        if movie_edges[["source_id", "target_id"]].isna().any().any():
            raise AssertionError("Edge endpoint was lost during node-ID remapping")
        normalized_nodes.append(movie_nodes)
        normalized_edges.append(movie_edges)
    return pd.concat(normalized_nodes), pd.concat(normalized_edges), clipped


source_path = find_source_submission()
print("Source graph:", source_path)
submission = pd.read_csv(source_path)
node_rows = submission[submission["row_type"].eq("node")].copy()
edge_rows = submission[submission["row_type"].eq("edge")].copy()
selected_edges = []
report = {}

for dataset in sorted(submission["dataset"].unique()):
    nodes = node_rows[node_rows["dataset"].eq(dataset)]
    edges = edge_rows[edge_rows["dataset"].eq(dataset)]
    selected = filter_edges(prepare_edge_geometry(nodes, edges))[edge_rows.columns]
    selected_edges.append(selected)
    report[dataset] = {
        "threshold_um": EDGE_THRESHOLD_UM,
        "input_edges": int(len(edges)),
        "output_edges": int(len(selected)),
    }

edge_rows = pd.concat(selected_edges, ignore_index=True)
node_rows, edge_rows, clipped_values = normalize_submission(node_rows, edge_rows)
columns = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
result = pd.concat([node_rows, edge_rows], ignore_index=True)[columns]
integer_columns = ["node_id", "t", "z", "y", "x", "source_id", "target_id"]
result[integer_columns] = result[integer_columns].astype(np.int64)

nodes = result[result["row_type"].eq("node")]
edges = result[result["row_type"].eq("edge")]
if set(result["dataset"].unique()) != set(submission["dataset"].unique()):
    raise AssertionError("Missing dataset")
if result.isna().any().any() or nodes.duplicated(["dataset", "node_id"]).any():
    raise AssertionError("Null value or duplicate node ID")
if (nodes["node_id"] <= 0).any() or (edges[["source_id", "target_id"]] <= 0).any().any():
    raise AssertionError("Node IDs and edge endpoints must be positive")
for column, (minimum, maximum) in BOUNDS.items():
    if not nodes[column].between(minimum, maximum).all():
        raise AssertionError(f"{column} is outside the valid image bounds")

node_keys = set(nodes[["dataset", "node_id"]].itertuples(index=False, name=None))
timestamps = {
    (dataset, node_id): int(frame)
    for dataset, node_id, frame in nodes[["dataset", "node_id", "t"]].itertuples(index=False)
}
for dataset, source, target in edges[["dataset", "source_id", "target_id"]].itertuples(index=False):
    if (dataset, source) not in node_keys or (dataset, target) not in node_keys:
        raise AssertionError("Dangling edge")
    if timestamps[(dataset, target)] != timestamps[(dataset, source)] + 1:
        raise AssertionError("Non-consecutive edge")
if edges.groupby(["dataset", "target_id"]).size().max() > 1:
    raise AssertionError("Multiple parents")
if edges.groupby(["dataset", "source_id"]).size().max() > 2:
    raise AssertionError("More than two children")

result.index.name = "id"
result.to_csv("submission.csv")
report["serialization"] = {"clipped_coordinate_values": clipped_values, "positive_contiguous_node_ids": True}
Path("calibration_report.json").write_text(json.dumps(report, indent=2))
print(f"Wrote submission.csv: {len(nodes):,} nodes, {len(edges):,} edges, {clipped_values} clipped coordinates")
