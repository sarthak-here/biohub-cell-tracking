"""Validate the completed BioHub baseline output against visible matching train graphs."""

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


def baseline_submission():
    candidates = [p for p in Path("/kaggle/input").rglob("submission.csv")
                  if "biohub-anisotropic-tracking-baseline" in str(p)]
    if not candidates:
        raise FileNotFoundError("Baseline submission output not attached")
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
    for t in np.intersect1d(nodes.t.unique(), np.unique(times)):
        p, mask = nodes[nodes.t.eq(t)], times == t
        pxyz, gxyz = p[["z", "y", "x"]].to_numpy(np.float32) * SCALE, xyz[mask] * SCALE
        if not len(pxyz) or not len(gxyz):
            continue
        distance = np.sqrt(((pxyz[:, None] - gxyz[None, :]) ** 2).sum(2))
        rows, cols = linear_sum_assignment(distance)
        pids, gids = p.node_id.to_numpy(np.int64), ids[mask]
        matched.update({int(pids[r]): int(gids[c]) for r, c in zip(rows, cols) if distance[r, c] <= 7.0})
    predicted_edges = {(matched[int(s)], matched[int(t)]) for s, t in
                       pred[pred.row_type.eq("edge")][["source_id", "target_id"]].itertuples(index=False)
                       if int(s) in matched and int(t) in matched}
    tp, fp, fn = len(predicted_edges & truth_edges), len(predicted_edges - truth_edges), len(truth_edges - predicted_edges)
    return {"edge_tp": tp, "edge_fp": fp, "edge_fn": fn,
            "edge_jaccard": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "node_recall": len(set(matched.values())) / len(ids) if len(ids) else 0.0}


root = competition_root()
submission = pd.read_csv(baseline_submission())
reports = []
for dataset in sorted(submission.dataset.unique()):
    geff = root / "train" / f"{dataset}.geff"
    if geff.exists():
        result = {"dataset": dataset, **score(submission[submission.dataset.eq(dataset)], geff)}
        reports.append(result)
        print(dataset, result)
if not reports:
    raise RuntimeError("No visible test movie matched training ground truth")
totals = {key: sum(int(row[key]) for row in reports) for key in ("edge_tp", "edge_fp", "edge_fn")}
denominator = sum(totals.values())
summary = {**totals, "edge_jaccard": totals["edge_tp"] / denominator if denominator else 0.0,
           "mean_node_recall": float(np.mean([row["node_recall"] for row in reports])), "movies": reports}
Path("validation.json").write_text(json.dumps(summary, indent=2))
print("LOCAL VALIDATION", summary)
