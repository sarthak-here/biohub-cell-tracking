"""Standalone BioHub cell detector and physical-motion tracker."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment


COMPETITION = "biohub-cell-tracking-during-development"
SCALE_UM = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
XY_DOWNSAMPLE = 4
DOG_SCALES_UM = ((1.5, 4.0), (2.2, 5.5))
PEAK_RADIUS_UM = 3.2
DOG_THRESHOLD = 0.045
MAX_PEAKS = 40_000
MAX_LINK_UM = 8.0
GAP_STEP_UM = 6.0
COLUMNS = [
    "id", "dataset", "row_type", "node_id", "t",
    "z", "y", "x", "source_id", "target_id",
]


@dataclass(frozen=True)
class ImageVolume:
    path: Path
    shape: tuple[int, int, int, int]
    dtype: np.dtype

    def frame(self, t: int) -> np.ndarray:
        chunk_path = self.path / "0" / "c" / str(t) / "0" / "0" / "0"
        compressed = chunk_path.read_bytes()
        try:
            import blosc2

            decoded = blosc2.decompress(compressed)
            array = np.frombuffer(decoded, dtype=self.dtype)
            if array.size == int(np.prod(self.shape[1:])):
                return array.reshape(self.shape[1:]).copy()
        except Exception:
            pass

        import zarr

        array = zarr.open(str(self.path / "0"), mode="r")
        return np.asarray(array[t])


def open_volume(path: Path) -> ImageVolume:
    metadata = json.loads((path / "0" / "zarr.json").read_text())
    shape = tuple(int(value) for value in metadata["shape"])
    if len(shape) != 4:
        raise ValueError(f"{path.name}: expected a TZYX array, got {shape}")
    return ImageVolume(path, shape, np.dtype(metadata["data_type"]))


def find_test_directory() -> Path:
    candidates = [
        Path(f"/kaggle/input/competitions/{COMPETITION}/test"),
        Path(f"/kaggle/input/{COMPETITION}/test"),
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.zarr")):
            return candidate
    for candidate in Path("/kaggle/input").rglob("test"):
        if candidate.is_dir() and any(candidate.glob("*.zarr")):
            return candidate
    raise FileNotFoundError("Could not locate the BioHub test directory")


def physical_footprint(radius_um: float, spacing_um: np.ndarray) -> np.ndarray:
    radius = np.maximum(1, np.ceil(radius_um / spacing_um).astype(int))
    zz, yy, xx = np.ogrid[
        -radius[0] : radius[0] + 1,
        -radius[1] : radius[1] + 1,
        -radius[2] : radius[2] + 1,
    ]
    distance_sq = (
        (zz * spacing_um[0]) ** 2
        + (yy * spacing_um[1]) ** 2
        + (xx * spacing_um[2]) ** 2
    )
    return distance_sq <= radius_um**2


def refine_centres(volume: np.ndarray, centres: np.ndarray) -> np.ndarray:
    result = centres.astype(np.float64, copy=True)
    limits = volume.shape
    for index, centre in enumerate(centres):
        z, y, x = np.rint(centre).astype(int)
        z0, z1 = max(0, z - 1), min(limits[0], z + 2)
        y0, y1 = max(0, y - 3), min(limits[1], y + 4)
        x0, x1 = max(0, x - 3), min(limits[2], x + 4)
        patch = volume[z0:z1, y0:y1, x0:x1].astype(np.float64)
        if not patch.size:
            continue
        weights = patch
        total = float(weights.sum())
        if total <= 0:
            continue
        grid = np.indices(patch.shape, dtype=np.float64)
        result[index] = [
            z0 + float((grid[0] * weights).sum() / total),
            y0 + float((grid[1] * weights).sum() / total),
            x0 + float((grid[2] * weights).sum() / total),
        ]
    return result


def detect_cells(volume: np.ndarray) -> np.ndarray:
    sampled = volume[:, ::XY_DOWNSAMPLE, ::XY_DOWNSAMPLE].astype(np.float32)
    low, high = np.percentile(sampled, [1.0, 99.7])
    if high <= low:
        return np.empty((0, 3), dtype=np.float64)
    normalized = np.clip((sampled - low) / (high - low), 0.0, None)
    spacing = SCALE_UM * np.array([1.0, XY_DOWNSAMPLE, XY_DOWNSAMPLE])

    response = None
    for small_um, large_um in DOG_SCALES_UM:
        current = gaussian_filter(normalized, small_um / spacing)
        current -= gaussian_filter(normalized, large_um / spacing)
        response = current if response is None else np.maximum(response, current)

    maxima = maximum_filter(
        response,
        footprint=physical_footprint(PEAK_RADIUS_UM, spacing),
        mode="nearest",
    )
    mask = (
        (response == maxima)
        & (response >= DOG_THRESHOLD)
        & (normalized >= np.percentile(normalized, 50.0))
    )
    centres = np.argwhere(mask)
    if not centres.size:
        return np.empty((0, 3), dtype=np.float64)
    order = np.argsort(response[mask])[::-1][:MAX_PEAKS]
    centres = centres[order].astype(np.float64)
    centres[:, 1:] *= XY_DOWNSAMPLE
    return refine_centres(volume, centres)


def link_frames(
    previous_ids: np.ndarray,
    previous_centres: np.ndarray,
    current_ids: np.ndarray,
    current_centres: np.ndarray,
) -> list[tuple[int, int]]:
    if not len(previous_ids) or not len(current_ids):
        return []
    previous_um = previous_centres * SCALE_UM
    current_um = current_centres * SCALE_UM
    distance = np.sqrt(
        ((previous_um[:, None, :] - current_um[None, :, :]) ** 2).sum(axis=2)
    )
    blocked_cost = MAX_LINK_UM * 1_000.0 + 1.0
    cost = np.where(distance <= MAX_LINK_UM, distance, blocked_cost)
    rows, columns = linear_sum_assignment(cost)
    return [
        (int(previous_ids[row]), int(current_ids[column]))
        for row, column in zip(rows, columns)
        if distance[row, column] <= MAX_LINK_UM
    ]


def validate_graph(
    nodes: list[dict[str, int]],
    edges: list[tuple[int, int]],
    shape: tuple[int, int, int, int],
) -> None:
    ids = [node["node_id"] for node in nodes]
    if not ids or ids != list(range(1, len(ids) + 1)):
        raise AssertionError("Node IDs must be nonempty, positive, and contiguous")
    times = {node["node_id"]: node["t"] for node in nodes}
    if len(times) != len(nodes):
        raise AssertionError("Duplicate node ID")
    for node in nodes:
        values = (node["t"], node["z"], node["y"], node["x"])
        if any(not 0 <= value < size for value, size in zip(values, shape)):
            raise AssertionError("Node coordinate outside the image")

    incoming: set[int] = set()
    outgoing: set[int] = set()
    seen: set[tuple[int, int]] = set()
    for source, target in edges:
        if (source, target) in seen:
            raise AssertionError("Duplicate edge")
        if source in outgoing or target in incoming:
            raise AssertionError("Multiple parents or children")
        if source not in times or target not in times:
            raise AssertionError("Dangling edge")
        if times[target] != times[source] + 1:
            raise AssertionError("Non-consecutive edge")
        seen.add((source, target))
        outgoing.add(source)
        incoming.add(target)


def close_single_frame_gaps(
    nodes: list[dict[str, int]],
    edges: list[tuple[int, int]],
    frame_count: int,
) -> tuple[list[dict[str, int]], list[tuple[int, int]], int]:
    by_id = {node["node_id"]: node for node in nodes}
    next_id = max(by_id, default=0) + 1
    added = 0
    for frame in range(frame_count - 2):
        incoming = {target for _, target in edges}
        outgoing = {source for source, _ in edges}
        ends = [
            node for node in nodes
            if node["t"] == frame and node["node_id"] not in outgoing
        ]
        starts = [
            node for node in nodes
            if node["t"] == frame + 2 and node["node_id"] not in incoming
        ]
        if not ends or not starts:
            continue
        end_xyz = np.array([[n["z"], n["y"], n["x"]] for n in ends], dtype=np.float64)
        start_xyz = np.array([[n["z"], n["y"], n["x"]] for n in starts], dtype=np.float64)
        distance = np.sqrt(
            (((end_xyz[:, None] - start_xyz[None, :]) * SCALE_UM) ** 2).sum(axis=2)
        )
        maximum = GAP_STEP_UM * 2.0
        cost = np.where(distance <= maximum, distance, maximum * 1_000.0 + 1.0)
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows, columns):
            if distance[row, column] > maximum:
                continue
            source = ends[row]
            target = starts[column]
            midpoint = np.rint((end_xyz[row] + start_xyz[column]) / 2.0).astype(int)
            nodes.append(
                {
                    "node_id": next_id,
                    "t": frame + 1,
                    "z": int(midpoint[0]),
                    "y": int(midpoint[1]),
                    "x": int(midpoint[2]),
                }
            )
            edges.extend(
                [(source["node_id"], next_id), (next_id, target["node_id"])]
            )
            by_id[next_id] = nodes[-1]
            next_id += 1
            added += 1
    return nodes, edges, added


def track_volume(
    image: ImageVolume,
) -> tuple[list[dict[str, int]], list[tuple[int, int]], dict[str, float]]:
    nodes: list[dict[str, int]] = []
    edges: list[tuple[int, int]] = []
    previous_ids = np.empty(0, dtype=np.int64)
    previous_centres = np.empty((0, 3), dtype=np.float64)
    next_id = 1

    for frame_index in range(image.shape[0]):
        volume = image.frame(frame_index)
        centres = detect_cells(volume)
        current_ids = np.arange(next_id, next_id + len(centres), dtype=np.int64)
        next_id += len(centres)
        edges.extend(link_frames(previous_ids, previous_centres, current_ids, centres))

        for node_id, centre in zip(current_ids, centres):
            position = np.clip(
                np.rint(centre).astype(np.int64),
                0,
                np.asarray(image.shape[1:]) - 1,
            )
            nodes.append(
                {
                    "node_id": int(node_id),
                    "t": frame_index,
                    "z": int(position[0]),
                    "y": int(position[1]),
                    "x": int(position[2]),
                }
            )
        previous_ids = current_ids
        previous_centres = centres

    nodes, edges, gap_nodes = close_single_frame_gaps(nodes, edges, image.shape[0])
    incident = {node_id for edge in edges for node_id in edge}
    if incident:
        nodes = [node for node in nodes if node["node_id"] in incident]
    mapping = {
        node["node_id"]: new_id for new_id, node in enumerate(nodes, start=1)
    }
    for node in nodes:
        node["node_id"] = mapping[node["node_id"]]
    edges = [(mapping[source], mapping[target]) for source, target in edges]
    validate_graph(nodes, edges, image.shape)
    return nodes, edges, {
        "frames": image.shape[0],
        "nodes": len(nodes),
        "edges": len(edges),
        "gap_nodes": gap_nodes,
        "edge_to_node_ratio": len(edges) / max(len(nodes), 1),
    }


def write_dataset(
    writer: csv.DictWriter,
    row_id: int,
    dataset: str,
    nodes: list[dict[str, int]],
    edges: list[tuple[int, int]],
) -> int:
    for node in nodes:
        writer.writerow(
            {
                "id": row_id,
                "dataset": dataset,
                "row_type": "node",
                **node,
                "source_id": -1,
                "target_id": -1,
            }
        )
        row_id += 1
    for source, target in edges:
        writer.writerow(
            {
                "id": row_id,
                "dataset": dataset,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": source,
                "target_id": target,
            }
        )
        row_id += 1
    return row_id


def main() -> None:
    test_directory = find_test_directory()
    datasets = sorted(test_directory.glob("*.zarr"))
    if not datasets:
        raise FileNotFoundError(f"No test data found in {test_directory}")

    report: dict[str, object] = {
        "method": "multi_scale_dog_hungarian",
        "test_directory": str(test_directory),
        "dataset_count": len(datasets),
        "max_link_distance_um": MAX_LINK_UM,
        "gap_step_um": GAP_STEP_UM,
        "datasets": {},
    }
    row_id = 0
    started = time.time()
    with Path("submission.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=COLUMNS)
        writer.writeheader()
        for index, dataset_path in enumerate(datasets, start=1):
            dataset_started = time.time()
            nodes, edges, stats = track_volume(open_volume(dataset_path))
            row_id = write_dataset(writer, row_id, dataset_path.stem, nodes, edges)
            stats["seconds"] = round(time.time() - dataset_started, 2)
            report["datasets"][dataset_path.stem] = stats
            print(
                f"[{index}/{len(datasets)}] {dataset_path.stem}: "
                f"{len(nodes):,} nodes, {len(edges):,} edges, "
                f"{stats['seconds']:.1f}s",
                flush=True,
            )

    report["rows"] = row_id
    report["seconds"] = round(time.time() - started, 2)
    Path("tracking_report.json").write_text(json.dumps(report, indent=2))
    print(
        f"Wrote submission.csv with {row_id:,} rows across {len(datasets)} datasets "
        f"in {report['seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
