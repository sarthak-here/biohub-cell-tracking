"""Validate the strongest public BioHub learned-model notebook on visible labels."""

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

SCALE = np.array([1.625, 0.40625, 0.40625], np.float32)
COMPETITION = "biohub-cell-tracking-during-development"


def competition_root():
    for path in (Path(f"/kaggle/input/competitions/{COMPETITION}"), Path(f"/kaggle/input/{COMPETITION}")):
        if (path / "train").is_dir():
            return path
    raise FileNotFoundError("Competition data not mounted")


def learned_submission():
    candidates = [p for p in Path("/kaggle/input").rglob("submission.csv")
                  if "biohub-cell-tracking-learned-graph-w-gap-recovery" in str(p)]
    if not candidates:
        raise FileNotFoundError("Public learned-model output not attached")
    print("Using", candidates[0])
    return candidates[0]


def arr(group, path):
    return np.asarray(group[path])


def score(pred, geff):
    group = zarr.open_group(str(geff), mode="r")
    ids = arr(group, "nodes/ids").astype(np.int64)
    times = arr(group, "nodes/props/t/values").astype(np.int64)
    xyz = np.column_stack([arr(group, f"nodes/props/{axis}/values") for axis in "zyx"]).astype(np.float32)
    truth_edges = {tuple(x) for x in arr(group, "edges/ids").astype(np.int64).reshape(-1, 2)}
    nodes, matched = pred[pred.row_type.eq("node")], {}
    matched_pred = set()
    for t in np.intersect1d(nodes.t.unique(), np.unique(times)):
        p, mask = nodes[nodes.t.eq(t)], times == t
        pxyz, gxyz = p[["z", "y", "x"]].to_numpy(np.float32) * SCALE, xyz[mask] * SCALE
        if not len(pxyz) or not len(gxyz):
            continue
        distance = np.sqrt(((pxyz[:, None] - gxyz[None, :]) ** 2).sum(2))
        rows, cols = linear_sum_assignment(distance)
        pids, gids = p.node_id.to_numpy(np.int64), ids[mask]
        for r, c in zip(rows, cols):
            if distance[r, c] <= 7.0:
                matched[int(pids[r])] = int(gids[c])
                matched_pred.add(int(pids[r]))
    edge_rows = pred[pred.row_type.eq("edge")][["source_id", "target_id"]]
    mapped_edges = {(matched[int(s)], matched[int(t)]) for s, t in edge_rows.itertuples(index=False)
                    if int(s) in matched and int(t) in matched}
    unmatched_edges = sum(int(s) not in matched or int(t) not in matched for s, t in edge_rows.itertuples(index=False))
    tp = len(mapped_edges & truth_edges)
    fp = len(mapped_edges - truth_edges) + unmatched_edges
    fn = len(truth_edges - mapped_edges)
    pred_nodes = len(nodes)
    return {"edge_tp": tp, "edge_fp": fp, "edge_fn": fn,
            "edge_jaccard": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "node_recall": len(set(matched.values())) / len(ids) if len(ids) else 0.0,
            "node_precision": len(matched_pred) / pred_nodes if pred_nodes else 0.0,
            "pred_nodes": pred_nodes, "truth_nodes": int(len(ids))}


root = competition_root()
submission = pd.read_csv(learned_submission())
reports = []
for dataset in sorted(submission.dataset.unique()):
    geff = root / "train" / f"{dataset}.geff"
    if geff.exists():
        result = {"dataset": dataset, **score(submission[submission.dataset.eq(dataset)], geff)}
        reports.append(result)
        print(dataset, result)
if not reports:
    raise RuntimeError("No visible test movie matched training labels")
totals = {key: sum(int(row[key]) for row in reports) for key in ("edge_tp", "edge_fp", "edge_fn")}
denominator = sum(totals.values())
summary = {**totals, "edge_jaccard": totals["edge_tp"] / denominator if denominator else 0.0,
           "mean_node_recall": float(np.mean([row["node_recall"] for row in reports])),
           "mean_node_precision": float(np.mean([row["node_precision"] for row in reports])), "movies": reports}
Path("public_validation.json").write_text(json.dumps(summary, indent=2))
print("PUBLIC MODEL VALIDATION", summary)
