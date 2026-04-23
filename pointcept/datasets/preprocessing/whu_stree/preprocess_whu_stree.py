"""
WHU-Stree 预处理脚本

功能：
1) 支持 split_samples 或原始 road 目录 + split txt 两种输入组织。
2) 将稀疏物种标签 remap 为连续训练 id。
3) 采用分块流式读取 PLY，降低超大点云预处理时的峰值内存。
"""

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


PLY_DTYPE_MAP = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def parse_binary_ply_header(ply_path):
    with ply_path.open("rb") as f:
        first_line = f.readline().decode("utf-8", errors="replace").strip()
        if first_line != "ply":
            raise ValueError(f"{ply_path} is not a valid PLY file")

        fmt_line = f.readline().decode("utf-8", errors="replace").strip()
        if "binary_little_endian" not in fmt_line:
            raise ValueError(
                f"{ply_path} format must be binary_little_endian, got: {fmt_line}"
            )

        vertex_count = None
        vertex_props = []
        in_vertex = False

        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while parsing header: {ply_path}")

            tokens = line.decode("utf-8", errors="replace").strip().split()
            if not tokens:
                continue

            if tokens[0] == "element":
                in_vertex = len(tokens) >= 3 and tokens[1] == "vertex"
                if in_vertex:
                    vertex_count = int(tokens[2])
                continue

            if tokens[0] == "property" and in_vertex:
                if len(tokens) != 3 or tokens[1] == "list":
                    raise ValueError(
                        f"Unsupported vertex property format in {ply_path}: {' '.join(tokens)}"
                    )
                p_type = tokens[1]
                p_name = tokens[2]
                if p_type not in PLY_DTYPE_MAP:
                    raise ValueError(
                        f"Unsupported PLY dtype {p_type} in {ply_path}."
                    )
                vertex_props.append((p_name, "<" + PLY_DTYPE_MAP[p_type]))
                continue

            if tokens[0] == "end_header":
                break

        if vertex_count is None or len(vertex_props) == 0:
            raise ValueError(f"Failed to parse vertex element from {ply_path}")

        offset = f.tell()

    return vertex_count, np.dtype(vertex_props), offset


def iter_vertex_chunks(ply_path, dtype, offset, vertex_count, chunk_points):
    with ply_path.open("rb") as f:
        f.seek(offset)
        remaining = vertex_count
        while remaining > 0:
            read_count = min(chunk_points, remaining)
            chunk = np.fromfile(f, dtype=dtype, count=read_count)
            if chunk.size == 0:
                break
            yield chunk
            remaining -= chunk.size


def collect_tasks_from_split_samples(dataset_root):
    tasks = []
    split_samples_root = dataset_root / "split_samples"
    for split in ["train", "val", "test"]:
        split_dir = split_samples_root / split
        if not split_dir.exists():
            continue
        for sample_dir in sorted(split_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            point_path = sample_dir / "point_cloud.ply"
            if not point_path.exists():
                continue
            reference_path = sample_dir / "reference_data.npy"
            tasks.append(
                {
                    "split": split,
                    "scene_name": sample_dir.name,
                    "point_path": str(point_path),
                    "reference_path": str(reference_path)
                    if reference_path.exists()
                    else None,
                }
            )
    return tasks


def collect_tasks_from_road_splits(dataset_root):
    split_files = {
        "train": dataset_root / "recommended_train_split.txt",
        "val": dataset_root / "recommended_val_split.txt",
        "test": dataset_root / "test_split.txt",
    }
    tasks = []

    for split, split_file in split_files.items():
        if not split_file.exists():
            continue
        with split_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                road_id = row["road_id"].strip()
                ply_id = row["ply_id"].strip()
                scene_name = f"{road_id}_{ply_id}"

                point_path = dataset_root / road_id / "PCD" / f"{ply_id}.ply"
                reference_path = dataset_root / "reference_data" / f"{scene_name}.npy"

                if not point_path.exists():
                    continue

                tasks.append(
                    {
                        "split": split,
                        "scene_name": scene_name,
                        "point_path": str(point_path),
                        "reference_path": str(reference_path)
                        if reference_path.exists()
                        else None,
                    }
                )

    return tasks


def resolve_tasks(dataset_root):
    split_sample_tasks = collect_tasks_from_split_samples(dataset_root)
    if len(split_sample_tasks) > 0:
        return split_sample_tasks, "split_samples"
    road_tasks = collect_tasks_from_road_splits(dataset_root)
    return road_tasks, "road_split_txt"


def collect_unique_labels(tasks, ignore_index, chunk_points):
    label_set = set()
    for task in tasks:
        if task["split"] not in {"train", "val"}:
            continue
        point_path = Path(task["point_path"])
        vertex_count, dtype, offset = parse_binary_ply_header(point_path)
        if "label" not in dtype.names:
            continue
        for chunk in iter_vertex_chunks(point_path, dtype, offset, vertex_count, chunk_points):
            labels = chunk["label"].astype(np.int32, copy=False)
            uniq = np.unique(labels)
            for lb in uniq:
                if int(lb) != int(ignore_index):
                    label_set.add(int(lb))

    if len(label_set) == 0:
        raise RuntimeError("No valid labels found from WHU-Stree train/val splits")
    return sorted(label_set)


def build_label_map(raw_labels, keep_sparse_label=False):
    if keep_sparse_label:
        return {int(lb): int(lb) for lb in raw_labels}
    return {int(lb): i for i, lb in enumerate(raw_labels)}


def remap_segment(raw_segment, sorted_raw_ids, mapped_ids, ignore_index):
    out = np.full(raw_segment.shape, ignore_index, dtype=np.int32)
    raw_segment = raw_segment.astype(np.int32, copy=False)

    mask = raw_segment != ignore_index
    if not np.any(mask):
        return out, 0

    values = raw_segment[mask]
    pos = np.searchsorted(sorted_raw_ids, values)
    valid = pos < sorted_raw_ids.shape[0]
    equal = np.zeros_like(valid, dtype=bool)
    equal[valid] = sorted_raw_ids[pos[valid]] == values[valid]
    valid = valid & equal

    mapped = np.full(values.shape, ignore_index, dtype=np.int32)
    mapped[valid] = mapped_ids[pos[valid]]
    out[mask] = mapped

    unknown_count = int((~valid).sum())
    return out, unknown_count


def process_one_scene(
    task,
    output_root,
    sorted_raw_ids,
    mapped_ids,
    ignore_index,
    chunk_points,
    overwrite,
    test_label_source,
    intensity_scale,
):
    point_path = Path(task["point_path"])
    split = task["split"]
    scene_name = task["scene_name"]
    reference_path = (
        Path(task["reference_path"]) if task.get("reference_path", None) else None
    )

    save_dir = Path(output_root) / split / scene_name
    save_dir.mkdir(parents=True, exist_ok=True)

    coord_path = save_dir / "coord.npy"
    strength_path = save_dir / "strength.npy"
    segment_path = save_dir / "segment.npy"
    instance_path = save_dir / "instance.npy"

    if (
        not overwrite
        and coord_path.exists()
        and strength_path.exists()
        and segment_path.exists()
        and instance_path.exists()
    ):
        return {
            "scene": scene_name,
            "split": split,
            "points": None,
            "skipped": True,
            "unknown_label_points": 0,
        }

    vertex_count, dtype, offset = parse_binary_ply_header(point_path)
    fields = set(dtype.names)

    required_fields = {"x", "y", "z", "intensity"}
    missing = required_fields - fields
    if missing:
        raise ValueError(
            f"{point_path} missing required fields: {sorted(list(missing))}. Found: {dtype.names}"
        )

    has_tree = "tree" in fields
    has_label = "label" in fields
    use_reference = (
        split == "test"
        and test_label_source == "reference"
        and reference_path is not None
        and reference_path.exists()
    )

    if split in {"train", "val"} and (not has_tree or not has_label):
        raise ValueError(
            f"{point_path} does not contain tree/label fields for {split} split"
        )

    reference_data = None
    if use_reference:
        reference_data = np.load(reference_path, mmap_mode="r")
        if reference_data.ndim != 2 or reference_data.shape[1] < 2:
            raise ValueError(
                f"Invalid reference_data shape in {reference_path}: {reference_data.shape}"
            )
        if reference_data.shape[0] != vertex_count:
            raise ValueError(
                f"reference_data length mismatch in {reference_path}. "
                f"expected {vertex_count}, got {reference_data.shape[0]}"
            )

    coord_mm = open_memmap(coord_path, mode="w+", dtype=np.float32, shape=(vertex_count, 3))
    strength_mm = open_memmap(strength_path, mode="w+", dtype=np.float32, shape=(vertex_count, 1))
    segment_mm = open_memmap(segment_path, mode="w+", dtype=np.int32, shape=(vertex_count,))
    instance_mm = open_memmap(instance_path, mode="w+", dtype=np.int32, shape=(vertex_count,))

    scale = max(float(intensity_scale), 1.0)
    cursor = 0
    unknown_label_points = 0

    for chunk in iter_vertex_chunks(point_path, dtype, offset, vertex_count, chunk_points):
        n = chunk.shape[0]
        end = cursor + n

        coord = np.stack([chunk["x"], chunk["y"], chunk["z"]], axis=1).astype(
            np.float32, copy=False
        )
        strength = (chunk["intensity"].astype(np.float32, copy=False) / scale).reshape(-1, 1)
        strength = np.clip(strength, 0.0, 1.0)

        if use_reference:
            inst_raw = reference_data[cursor:end, 0].astype(np.int32, copy=False)
            seg_raw = reference_data[cursor:end, 1].astype(np.int32, copy=False)
        else:
            if has_tree:
                # tree 在原始 PLY 中通常是 float 编码，这里转回整数 id。
                inst_raw = np.rint(chunk["tree"]).astype(np.int32)
            else:
                inst_raw = np.full((n,), ignore_index, dtype=np.int32)

            if has_label:
                seg_raw = chunk["label"].astype(np.int32, copy=False)
            else:
                seg_raw = np.full((n,), ignore_index, dtype=np.int32)

        seg, unknown_count = remap_segment(seg_raw, sorted_raw_ids, mapped_ids, ignore_index)
        unknown_label_points += int(unknown_count)

        inst = inst_raw.copy()
        inst[seg == ignore_index] = ignore_index
        inst[inst <= 0] = ignore_index

        coord_mm[cursor:end] = coord
        strength_mm[cursor:end] = strength
        segment_mm[cursor:end] = seg
        instance_mm[cursor:end] = inst
        cursor = end

    if cursor != vertex_count:
        raise RuntimeError(
            f"Point count mismatch for {point_path}. header={vertex_count}, read={cursor}"
        )

    del coord_mm
    del strength_mm
    del segment_mm
    del instance_mm

    return {
        "scene": scene_name,
        "split": split,
        "points": vertex_count,
        "skipped": False,
        "unknown_label_points": unknown_label_points,
    }


def main():
    parser = argparse.ArgumentParser("Preprocess WHU-Stree into Pointcept format")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="WHU-Stree root, e.g. /data/tpy/raw-datasets/WHU-Stree/whu-stree-nj",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output root, e.g. data/whu_stree",
    )
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=-1,
        help="Ignore label value used by Pointcept",
    )
    parser.add_argument(
        "--chunk_points",
        type=int,
        default=1_000_000,
        help="Streaming chunk size (points per read), smaller means lower peak memory",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of worker processes. Keep 1 if memory is tight.",
    )
    parser.add_argument(
        "--max_scenes",
        type=int,
        default=0,
        help="Only preprocess the first N scenes for debug. 0 means all scenes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing preprocessed files",
    )
    parser.add_argument(
        "--keep_sparse_label",
        action="store_true",
        help="Keep sparse species ids instead of remapping to contiguous train ids",
    )
    parser.add_argument(
        "--test_label_source",
        type=str,
        default="reference",
        choices=["reference", "none"],
        help="How to build test labels: reference uses reference_data.npy, none fills ignore",
    )
    parser.add_argument(
        "--intensity_scale",
        type=float,
        default=65535.0,
        help="Normalization divisor for intensity -> strength",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks, source_mode = resolve_tasks(dataset_root)
    if len(tasks) == 0:
        raise RuntimeError(f"No WHU scenes found under {dataset_root}")

    if args.max_scenes > 0:
        tasks = tasks[: args.max_scenes]

    print(f"[WHU-Stree] Source mode: {source_mode}")
    print(f"[WHU-Stree] Total scenes: {len(tasks)}")

    raw_labels = collect_unique_labels(
        tasks, ignore_index=int(args.ignore_index), chunk_points=int(args.chunk_points)
    )
    label_map = build_label_map(
        raw_labels=raw_labels, keep_sparse_label=bool(args.keep_sparse_label)
    )
    sorted_raw_ids = np.array(sorted(label_map.keys()), dtype=np.int32)
    mapped_ids = np.array([label_map[int(x)] for x in sorted_raw_ids], dtype=np.int32)

    worker_kwargs = dict(
        output_root=str(output_root),
        sorted_raw_ids=sorted_raw_ids,
        mapped_ids=mapped_ids,
        ignore_index=int(args.ignore_index),
        chunk_points=int(args.chunk_points),
        overwrite=bool(args.overwrite),
        test_label_source=str(args.test_label_source),
        intensity_scale=float(args.intensity_scale),
    )

    results = []
    if args.num_workers <= 1:
        for task in tasks:
            results.append(process_one_scene(task=task, **worker_kwargs))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(process_one_scene, task=task, **worker_kwargs)
                for task in tasks
            ]
            for future in as_completed(futures):
                results.append(future.result())

    split_counter = {"train": 0, "val": 0, "test": 0}
    point_counter = {"train": 0, "val": 0, "test": 0}
    unknown_counter = {"train": 0, "val": 0, "test": 0}
    skipped = 0
    for r in results:
        split_counter[r["split"]] += 1
        if r["points"] is not None:
            point_counter[r["split"]] += int(r["points"])
        unknown_counter[r["split"]] += int(r.get("unknown_label_points", 0))
        if r.get("skipped", False):
            skipped += 1

    meta = {
        "dataset": "WHU-Stree",
        "source_mode": source_mode,
        "ignore_index": int(args.ignore_index),
        "keep_sparse_label": bool(args.keep_sparse_label),
        "test_label_source": str(args.test_label_source),
        "intensity_scale": float(args.intensity_scale),
        "label_mapping_raw_to_trainid": label_map,
        "train_class_count": int(len(label_map)),
        "split_scene_count": split_counter,
        "split_point_count": point_counter,
        "split_unknown_label_points": unknown_counter,
        "skipped_scene_count": skipped,
        "chunk_points": int(args.chunk_points),
    }
    with (output_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[WHU-Stree] Done")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
