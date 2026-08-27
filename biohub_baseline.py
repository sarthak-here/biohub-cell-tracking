"""Rules-compliant BioHub baseline: 3D local-maxima detection and temporal linking."""

from __future__ import annotations

import json
from pathlib import Path

import blosc2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

COMPETITION = "biohub-cell-tracking-during-development"
VOXEL_SIZE_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float32)
XY_DOWNSAMPLE = 2
MAX_LINK_DISTANCE_UM = 8.0
PEAK_PERCENTILE = 99.7
MAX_NODES_PER_FRAME = 2600


def competition_root() -> Path:
    for path in (Path(f"/kaggle/input/competitions/{COMPETITION}"), Path(f"/kaggle/input/{COMPETITION}")):
        if (path / "test").is_dir():
            return path
    raise FileNotFoundError("BioHub competition data was not mounted by Kaggle.")


def load_frame(zarr_path: Path, frame: int, shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    chunk = zarr_path / "0" / "c" / str(frame) / "0" / "0" / "0"
    return np.frombuffer(blosc2.decompress(chunk.read_bytes()), dtype=dtype).reshape(shape[1:])


def detect_centres(volume: np.ndarray) -> np.ndarray:
    """Find bright 3D local maxima; preserve the coarser Z resolution."""
    reduced = volume[:, ::XY_DOWNSAMPLE, ::XY_DOWNSAMPLE].astype(np.float32, copy=False)
    smooth = gaussian_filter(reduced, sigma=(0.8, 1.25, 1.25))
    threshold = np.percentile(smooth, PEAK_PERCENTILE)
    points = np.argwhere((smooth == maximum_filter(smooth, size=(3, 5, 5))) & (smooth >= threshold))
    if len(points) > MAX_NODES_PER_FRAME:
        intensity = smooth[tuple(points.T)]
        points = points[np.argpartition(intensity, -MAX_NODES_PER_FRAME)[-MAX_NODES_PER_FRAME:]]
    points = points.astype(np.int32, copy=False)
    points[:, 1:] *= XY_DOWNSAMPLE
    return points


def link_frames(previous: dict[int, np.ndarray], current: dict[int, np.ndarray]) -> list[tuple[int, int]]:
    """Distance-gated, one-to-one links in microns (not raw anisotropic voxels)."""
    if not previous or not current:
        return []
    parent_ids = np.fromiter(previous.keys(), dtype=np.int64)
    child_ids = np.fromiter(current.keys(), dtype=np.int64)
    parents = np.stack([previous[node_id] for node_id in parent_ids]) * VOXEL_SIZE_UM
    children = np.stack([current[node_id] for node_id in child_ids]) * VOXEL_SIZE_UM
    distances, parent_idx = cKDTree(parents).query(children, distance_upper_bound=MAX_LINK_DISTANCE_UM)
    proposals = sorted(
        (float(distance), int(parent_ids[idx]), int(child_ids[child]))
        for child, (distance, idx) in enumerate(zip(distances, parent_idx))
        if np.isfinite(distance) and idx < len(parent_ids)
    )
    used: set[int] = set()
    links: list[tuple[int, int]] = []
    for _, parent_id, child_id in proposals:
        if parent_id not in used:
            used.add(parent_id)
            links.append((parent_id, child_id))
    return links


def node_row(dataset: str, node_id: int, frame: int, point: np.ndarray) -> dict[str, object]:
    return {"dataset": dataset, "row_type": "node", "node_id": node_id, "t": frame,
            "z": int(point[0]), "y": int(point[1]), "x": int(point[2]),
            "source_id": -1, "target_id": -1}


def edge_row(dataset: str, source_id: int, target_id: int) -> dict[str, object]:
    return {"dataset": dataset, "row_type": "edge", "node_id": -1, "t": -1,
            "z": -1, "y": -1, "x": -1, "source_id": source_id, "target_id": target_id}


def track_movie(movie: Path) -> list[dict[str, object]]:
    metadata = json.loads((movie / "0" / "zarr.json").read_text())
    shape, dtype = tuple(metadata["shape"]), np.dtype(metadata["data_type"])
    rows: list[dict[str, object]] = []
    previous: dict[int, np.ndarray] = {}
    next_id = 1
    for frame in range(shape[0]):
        points = detect_centres(load_frame(movie, frame, shape, dtype))
        current = {next_id + index: point for index, point in enumerate(points)}
        rows.extend(node_row(movie.stem, node_id, frame, point) for node_id, point in current.items())
        rows.extend(edge_row(movie.stem, source, target) for source, target in link_frames(previous, current))
        next_id += len(current)
        previous = current
        print(f"{movie.stem}: frame {frame + 1}/{shape[0]}, nodes={len(current)}")
    return rows


def validate_submission(submission: pd.DataFrame, datasets: set[str]) -> None:
    required = {"dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"}
    if missing := required.difference(submission.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if set(submission.dataset.unique()) != datasets:
        raise ValueError("Not all mounted test movies are represented.")
    nodes = submission[submission.row_type.eq("node")][["dataset", "node_id", "t"]]
    if nodes.duplicated(["dataset", "node_id"]).any():
        raise ValueError("Node IDs must be unique within each movie.")
    timestamps = {(dataset, node): frame for dataset, node, frame in nodes.itertuples(index=False)}
    for dataset, source, target in submission[submission.row_type.eq("edge")][["dataset", "source_id", "target_id"]].itertuples(index=False):
        if (dataset, source) not in timestamps or (dataset, target) not in timestamps:
            raise ValueError("Edge endpoint is missing.")
        if timestamps[(dataset, target)] != timestamps[(dataset, source)] + 1:
            raise ValueError("Edges must link consecutive frames.")


def main() -> None:
    movies = sorted((competition_root() / "test").glob("*.zarr"))
    if not movies:
        raise FileNotFoundError("No test movies found.")
    rows = [row for movie in movies for row in track_movie(movie)]
    submission = pd.DataFrame(rows)
    validate_submission(submission, {movie.stem for movie in movies})
    submission.index.name = "id"
    submission.to_csv("submission.csv")
    print(f"Wrote submission.csv: {len(submission):,} rows across {len(movies)} movies.")


if __name__ == "__main__":
    main()
