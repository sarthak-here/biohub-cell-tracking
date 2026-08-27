"""Calibrate a public pretrained BioHub tracking graph using physical motion."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

try:
    import zarr
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "zarr"])
    import zarr

COMPETITION = "biohub-cell-tracking-during-development"
SCALE = np.array([1.625, 0.40625, 0.40625], np.float32)
THRESHOLDS = [6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 1000.0]


def find_competition():
    choices = [Path(f"/kaggle/input/competitions/{COMPETITION}"), Path(f"/kaggle/input/{COMPETITION}")]
    return next((path for path in choices if (path / "train").is_dir()), None)


def find_source():
    found = [path for path in Path("/kaggle/input").rglob("submission.csv")
             if "biohub-cell-tracking-learned-graph-w-gap-recovery" in str(path)]
    return found[0] if found else None


def read_array(group, key):
    return np.asarray(group[key])


def read_truth(path):
    group = zarr.open_group(str(path), mode="r")
    node_ids = read_array(group, "nodes/ids").astype(np.int64)
    times = read_array(group, "nodes/props/t/values").astype(np.int64)
    points = np.column_stack([read_array(group, f"nodes/props/{axis}/values") for axis in "zyx"]).astype(np.float32)
    edges = {tuple(row) for row in read_array(group, "edges/ids").astype(np.int64).reshape(-1, 2)}
    return node_ids, times, points, edges


def match_visible_nodes(nodes, truth_ids, truth_times, truth_points):
    matched = {}
    for frame in np.intersect1d(nodes["t"].unique(), np.unique(truth_times)):
        pred = nodes[nodes["t"].eq(frame)]
        mask = truth_times == frame
        pred_xyz = pred[["z", "y", "x"]].to_numpy(np.float32) * SCALE
        true_xyz = truth_points[mask] * SCALE
        if not len(pred_xyz) or not len(true_xyz):
            continue
        distance = np.sqrt(((pred_xyz[:, None] - true_xyz[None, :]) ** 2).sum(axis=2))
        rows, columns = linear_sum_assignment(distance)
        pred_ids = pred["node_id"].to_numpy(np.int64)
        visible_ids = truth_ids[mask]
        for row, column in zip(rows, columns):
            if distance[row, column] <= 7.0:
                matched[int(pred_ids[row])] = int(visible_ids[column])
    return matched


def prepare_edge_geometry(nodes, edges):
    source = nodes.set_index("node_id")[["t", "z", "y", "x"]].add_prefix("source_")
    target = nodes.set_index("node_id")[["t", "z", "y", "x"]].add_prefix("target_")
    work = edges.join(source, on="source_id").join(target, on="target_id").dropna()
    work = work[work["target_t"].eq(work["source_t"] + 1)].copy()
    delta = (work[["target_z", "target_y", "target_x"]].to_numpy(np.float32)
             - work[["source_z", "source_y", "source_x"]].to_numpy(np.float32)) * SCALE
    work["distance_um"] = np.sqrt((delta * delta).sum(axis=1))
    return work


def filter_edges(geometry, threshold):
    work = geometry[geometry["distance_um"].le(threshold)].sort_values("distance_um")
    work = work.drop_duplicates("target_id", keep="first")
    work = work.groupby("source_id", sort=False, group_keys=False).head(2)
    return work


def edge_score(edges, matched, truth_edges):
    mapped = {(matched[int(source)], matched[int(target)])
              for source, target in edges[["source_id", "target_id"]].itertuples(index=False)
              if int(source) in matched and int(target) in matched}
    true_positive = len(mapped & truth_edges)
    false_positive = len(mapped - truth_edges)
    false_negative = len(truth_edges - mapped)
    denominator = true_positive + false_positive + false_negative
    return {"tp": true_positive, "fp": false_positive, "fn": false_negative,
            "edge_jaccard": true_positive / denominator if denominator else 0.0}


root, source_path = find_competition(), find_source()
if root is None:
    raise FileNotFoundError("BioHub competition data was not mounted")
if source_path is None:
    raise FileNotFoundError("Public learned tracker output was not attached")
print("Source:", source_path)

submission = pd.read_csv(source_path)
node_rows = submission[submission["row_type"].eq("node")].copy()
edge_rows = submission[submission["row_type"].eq("edge")].copy()
chosen_edge_frames, report = [], {}

for dataset in sorted(submission["dataset"].unique()):
    nodes = node_rows[node_rows["dataset"].eq(dataset)]
    edges = edge_rows[edge_rows["dataset"].eq(dataset)]
    geometry = prepare_edge_geometry(nodes, edges)
    truth_path = root / "train" / f"{dataset}.geff"
    if not truth_path.exists():
        threshold, validation = 14.0, None
    else:
        truth_ids, truth_times, truth_points, truth_edges = read_truth(truth_path)
        matched = match_visible_nodes(nodes, truth_ids, truth_times, truth_points)
        grid = []
        for candidate_threshold in THRESHOLDS:
            candidate_edges = filter_edges(geometry, candidate_threshold)
            grid.append({"threshold_um": candidate_threshold, "edges": int(len(candidate_edges)),
                         **edge_score(candidate_edges, matched, truth_edges)})
        validation = max(grid, key=lambda row: (row["edge_jaccard"], row["tp"], -row["threshold_um"]))
        threshold = float(validation["threshold_um"])
        report[dataset] = {"threshold_um": threshold,
                           "node_recall": len(set(matched.values())) / len(truth_ids) if len(truth_ids) else 0.0,
                           "validation": validation, "grid": grid}
    selected = filter_edges(geometry, threshold)[edge_rows.columns]
    chosen_edge_frames.append(selected)
    report.setdefault(dataset, {"threshold_um": threshold, "validation": validation})
    print(dataset, "threshold", threshold, "edges", len(edges), "->", len(selected))

columns = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
result = pd.concat([node_rows, *chosen_edge_frames], ignore_index=True)[columns]
if set(result["dataset"].unique()) != set(submission["dataset"].unique()):
    raise AssertionError("Missing dataset after calibration")
nodes = result[result["row_type"].eq("node")]
if nodes.duplicated(["dataset", "node_id"]).any():
    raise AssertionError("Duplicate node IDs")
node_keys = set(nodes[["dataset", "node_id"]].itertuples(index=False, name=None))
for dataset, source, target in result[result["row_type"].eq("edge")][["dataset", "source_id", "target_id"]].itertuples(index=False):
    if (dataset, source) not in node_keys or (dataset, target) not in node_keys:
        raise AssertionError("Dangling edge")

result.index.name = "id"
result.to_csv("submission.csv")
Path("adaptive_validation.json").write_text(json.dumps(report, indent=2))
print(f"Wrote submission.csv: {len(nodes):,} nodes, {len(result) - len(nodes):,} edges")
