"""Score the standalone tracker against matching visible training graphs."""

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


SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
COMPETITION = "biohub-cell-tracking-during-development"
SOURCE_SLUG = "biohub-standalone-physical-tracker"


def competition_root() -> Path:
    candidates = [
        Path(f"/kaggle/input/competitions/{COMPETITION}"),
        Path(f"/kaggle/input/{COMPETITION}"),
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir():
            return candidate
    raise FileNotFoundError("Competition training data is not mounted")


def tracker_submission() -> Path:
    candidates = [
        path
        for path in Path("/kaggle/input").rglob("submission.csv")
        if SOURCE_SLUG in str(path)
    ]
    if not candidates:
        raise FileNotFoundError("Standalone tracker output is not attached")
    return candidates[0]


def array(group, path: str) -> np.ndarray:
    return np.asarray(group[path])


def estimated_node_count(geff_path: Path) -> float:
    metadata = json.loads((geff_path / "zarr.json").read_text())
    geff = metadata.get("attributes", {}).get("geff", {})
    value = (geff.get("extra", {}) or {}).get("estimated_number_of_nodes")
    return float(value) if value is not None else float("nan")


def score(prediction: pd.DataFrame, geff_path: Path) -> dict[str, float]:
    ground_truth = zarr.open_group(str(geff_path), mode="r")
    truth_ids = array(ground_truth, "nodes/ids").astype(np.int64)
    truth_times = array(ground_truth, "nodes/props/t/values").astype(np.int64)
    truth_xyz = np.column_stack(
        [array(ground_truth, f"nodes/props/{axis}/values") for axis in "zyx"]
    ).astype(np.float64)
    truth_edges = {
        tuple(edge)
        for edge in array(ground_truth, "edges/ids").astype(np.int64).reshape(-1, 2)
    }
    truth_sources = {source for source, _ in truth_edges}
    truth_targets = {target for _, target in truth_edges}

    nodes = prediction[prediction.row_type.eq("node")]
    matches: dict[int, int] = {}
    for frame in np.intersect1d(nodes.t.unique(), np.unique(truth_times)):
        predicted = nodes[nodes.t.eq(frame)]
        truth_mask = truth_times == frame
        predicted_xyz = predicted[["z", "y", "x"]].to_numpy(np.float64) * SCALE
        frame_truth_xyz = truth_xyz[truth_mask] * SCALE
        if not len(predicted_xyz) or not len(frame_truth_xyz):
            continue
        distance = np.sqrt(
            ((predicted_xyz[:, None] - frame_truth_xyz[None, :]) ** 2).sum(axis=2)
        )
        blocked = np.where(distance <= 7.0, distance, 7_001.0)
        rows, columns = linear_sum_assignment(blocked)
        predicted_ids = predicted.node_id.to_numpy(np.int64)
        frame_truth_ids = truth_ids[truth_mask]
        for row, column in zip(rows, columns):
            if distance[row, column] <= 7.0:
                matches[int(predicted_ids[row])] = int(frame_truth_ids[column])

    matched_truth_edges: set[tuple[int, int]] = set()
    false_positive_edges = 0
    edges = prediction[prediction.row_type.eq("edge")]
    for source, target in edges[["source_id", "target_id"]].itertuples(index=False):
        matched_source = matches.get(int(source))
        matched_target = matches.get(int(target))
        matched_edge = (
            matched_source is not None
            and matched_target is not None
            and (matched_source, matched_target) in truth_edges
        )
        if matched_edge:
            matched_truth_edges.add((matched_source, matched_target))
            continue
        source_is_evaluable = matched_source is not None and matched_source in truth_sources
        target_is_evaluable = matched_target is not None and matched_target in truth_targets
        if source_is_evaluable or target_is_evaluable:
            false_positive_edges += 1

    edge_tp = len(matched_truth_edges)
    edge_fp = false_positive_edges
    edge_fn = len(truth_edges) - edge_tp
    denominator = edge_tp + edge_fp + edge_fn
    edge_jaccard = edge_tp / denominator if denominator else 0.0
    estimated_total = estimated_node_count(geff_path)
    node_ratio = (
        (len(nodes) - estimated_total) / estimated_total
        if estimated_total > 0
        else float("nan")
    )
    adjusted = max(0.0, edge_jaccard * (1.0 - 0.1 * node_ratio))
    division_count = sum(
        count == 2
        for count in pd.Series([source for source, _ in truth_edges]).value_counts()
    )
    return {
        "edge_tp": edge_tp,
        "edge_fp": edge_fp,
        "edge_fn": edge_fn,
        "edge_jaccard": edge_jaccard,
        "adjusted_edge_jaccard": adjusted,
        "node_recall": len(set(matches.values())) / max(len(truth_ids), 1),
        "predicted_nodes": len(nodes),
        "estimated_nodes": estimated_total,
        "division_fn": division_count,
        "weight": denominator,
    }


root = competition_root()
submission_path = tracker_submission()
submission = pd.read_csv(submission_path)
reports = []
for dataset in sorted(submission.dataset.unique()):
    geff_path = root / "train" / f"{dataset}.geff"
    if geff_path.exists():
        result = {"dataset": dataset, **score(submission[submission.dataset.eq(dataset)], geff_path)}
        reports.append(result)
        print(dataset, result)
if not reports:
    raise RuntimeError("No visible test movie matched a training graph")

edge_tp = sum(row["edge_tp"] for row in reports)
edge_fp = sum(row["edge_fp"] for row in reports)
edge_fn = sum(row["edge_fn"] for row in reports)
total_weight = sum(row["weight"] for row in reports)
summary = {
    "movies": reports,
    "edge_tp": edge_tp,
    "edge_fp": edge_fp,
    "edge_fn": edge_fn,
    "edge_jaccard": edge_tp / max(edge_tp + edge_fp + edge_fn, 1),
    "adjusted_edge_jaccard": sum(
        row["weight"] * row["adjusted_edge_jaccard"] for row in reports
    ) / max(total_weight, 1),
    "mean_node_recall": float(np.mean([row["node_recall"] for row in reports])),
    "division_jaccard": 0.0,
}
summary["score"] = summary["adjusted_edge_jaccard"]
Path("validation.json").write_text(json.dumps(summary, indent=2))
print("VISIBLE VALIDATION", summary)
