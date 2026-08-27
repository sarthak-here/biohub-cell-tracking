"""Diagnose per-movie bright/dark peak detection on BioHub labeled frames."""

import json
import subprocess
import sys
from pathlib import Path

import blosc2
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
from scipy.optimize import linear_sum_assignment

try:
    import zarr
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "zarr"])
    import zarr

COMPETITION = "biohub-cell-tracking-during-development"
SCALE = np.array([1.625, 0.40625, 0.40625], np.float32)
XY_DOWNSAMPLE = 2
PERCENTILES = [90.0, 95.0, 97.0, 98.0, 99.0, 99.5, 99.7]


def root():
    for path in (Path(f"/kaggle/input/competitions/{COMPETITION}"), Path(f"/kaggle/input/{COMPETITION}")):
        if (path / "test").is_dir():
            return path
    raise FileNotFoundError("Competition data not mounted")


def array(group, path):
    return np.asarray(group[path])


def load_frame(movie, frame, shape, dtype):
    chunk = movie / "0" / "c" / str(frame) / "0" / "0" / "0"
    return np.frombuffer(blosc2.decompress(chunk.read_bytes()), dtype=dtype).reshape(shape[1:])


def match(pred, truth):
    if not len(pred) or not len(truth):
        return 0, len(pred), len(truth)
    distances = np.sqrt((((pred[:, None] - truth[None, :]) * SCALE) ** 2).sum(2))
    rows, cols = linear_sum_assignment(distances)
    tp = sum(distances[r, c] <= 7.0 for r, c in zip(rows, cols))
    return int(tp), int(len(pred) - tp), int(len(truth) - tp)


def candidates(smooth, percentile, polarity):
    if polarity == "bright":
        mask = (smooth == maximum_filter(smooth, size=(3, 5, 5))) & (smooth >= np.percentile(smooth, percentile))
    else:
        mask = (smooth == minimum_filter(smooth, size=(3, 5, 5))) & (smooth <= np.percentile(smooth, 100.0 - percentile))
    points = np.argwhere(mask).astype(np.float32)
    points[:, 1:] *= XY_DOWNSAMPLE
    return points


base = root()
report = {}
for movie in sorted((base / "test").glob("*.zarr")):
    geff = base / "train" / f"{movie.stem}.geff"
    if not geff.exists():
        continue
    group = zarr.open_group(str(geff), mode="r")
    times = array(group, "nodes/props/t/values").astype(np.int64)
    truth_xyz = np.column_stack([array(group, f"nodes/props/{axis}/values") for axis in "zyx"]).astype(np.float32)
    metadata = json.loads((movie / "0" / "zarr.json").read_text())
    shape, dtype = tuple(metadata["shape"]), np.dtype(metadata["data_type"])
    totals = {(polarity, p): [0, 0, 0] for polarity in ("bright", "dark") for p in PERCENTILES}
    labelled_frames = [int(t) for t in np.unique(times) if 0 <= int(t) < shape[0]]
    for frame in labelled_frames:
        volume = load_frame(movie, frame, shape, dtype)
        reduced = volume[:, ::XY_DOWNSAMPLE, ::XY_DOWNSAMPLE].astype(np.float32, copy=False)
        smooth = gaussian_filter(reduced, sigma=(0.8, 1.25, 1.25))
        truth = truth_xyz[times == frame]
        for polarity in ("bright", "dark"):
            for percentile in PERCENTILES:
                result = match(candidates(smooth, percentile, polarity), truth)
                totals[(polarity, percentile)] = [a + b for a, b in zip(totals[(polarity, percentile)], result)]
    rows = []
    for (polarity, percentile), (tp, fp, fn) in totals.items():
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"polarity": polarity, "percentile": percentile, "tp": tp, "fp": fp, "fn": fn,
                     "precision": precision, "recall": recall, "f1": f1})
    rows.sort(key=lambda row: row["f1"], reverse=True)
    report[movie.stem] = {"labelled_frames": len(labelled_frames), "truth_nodes": int(len(truth_xyz)),
                          "best": rows[0], "grid": rows}
    print(movie.stem, report[movie.stem]["best"])

Path("detection_diagnostic.json").write_text(json.dumps(report, indent=2))
print("Wrote detection_diagnostic.json")
