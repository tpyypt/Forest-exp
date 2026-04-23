"""
FOR-InstanceV2 预处理脚本

目标：将原始 PLY 转为 Pointcept 标准目录结构，避免超大场景一次性读入导致 OOM。
"""

import argparse
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


def parse_semantic_map(semantic_map_str):
    semantic_map = {}
    for item in semantic_map_str.split(","):
        item = item.strip()
        if not item:
            continue
        src, dst = item.split(":")
        semantic_map[int(src)] = int(dst)
    if not semantic_map:
        raise ValueError("semantic_map is empty. Example: 1:0,2:1,3:2")
    return semantic_map


def parse_int_list(values):
    if values is None or values.strip() == "":
        return np.array([], dtype=np.int32)
    return np.array([int(v.strip()) for v in values.split(",")], dtype=np.int32)


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
                # 仅支持标量属性。当前两个数据集均符合该约束。
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


def remap_segment(raw_segment, semantic_map, ignore_index):
    remapped = np.full(raw_segment.shape, ignore_index, dtype=np.int32)
    raw_segment = raw_segment.astype(np.int32, copy=False)
    for src, dst in semantic_map.items():
        remapped[raw_segment == src] = dst
    return remapped


def build_tasks(dataset_root):
    tasks = []
    train_val_root = dataset_root / "train_val_data"
    test_root = dataset_root / "test_data"

    for ply_path in sorted(train_val_root.glob("*.ply")):
        stem = ply_path.stem
        if stem.endswith("_train"):
            split = "train"
        elif stem.endswith("_val"):
            split = "val"
        else:
            continue
        tasks.append(
            {
                "split": split,
                "scene_name": stem,
                "point_path": str(ply_path),
            }
        )

    for ply_path in sorted(test_root.glob("*.ply")):
        tasks.append(
            {
                "split": "test",
                "scene_name": ply_path.stem,
                "point_path": str(ply_path),
            }
        )

    return tasks


def process_one_scene(
    task,
    output_root,
    semantic_map,
    ignore_index,
    tree_segment_ids,
    chunk_points,
    overwrite,
):
    point_path = Path(task["point_path"])
    split = task["split"]
    scene_name = task["scene_name"]

    save_dir = Path(output_root) / split / scene_name
    save_dir.mkdir(parents=True, exist_ok=True)

    coord_path = save_dir / "coord.npy"
    segment_path = save_dir / "segment.npy"
    instance_path = save_dir / "instance.npy"

    if (
        not overwrite
        and coord_path.exists()
        and segment_path.exists()
        and instance_path.exists()
    ):
        return {"scene": scene_name, "split": split, "points": None, "skipped": True}

    vertex_count, dtype, offset = parse_binary_ply_header(point_path)

    required_fields = {"x", "y", "z", "semantic_seg", "treeID"}
    actual_fields = set(dtype.names)
    missing = required_fields - actual_fields
    if missing:
        raise ValueError(
            f"{point_path} missing required fields: {sorted(list(missing))}. "
            f"Found: {dtype.names}"
        )

    coord_mm = open_memmap(coord_path, mode="w+", dtype=np.float32, shape=(vertex_count, 3))
    segment_mm = open_memmap(segment_path, mode="w+", dtype=np.int32, shape=(vertex_count,))
    instance_mm = open_memmap(instance_path, mode="w+", dtype=np.int32, shape=(vertex_count,))

    cursor = 0
    for chunk in iter_vertex_chunks(point_path, dtype, offset, vertex_count, chunk_points):
        n = chunk.shape[0]
        end = cursor + n

        coord = np.stack([chunk["x"], chunk["y"], chunk["z"]], axis=1).astype(
            np.float32, copy=False
        )
        segment = remap_segment(chunk["semantic_seg"], semantic_map, ignore_index)

        instance = chunk["treeID"].astype(np.int32, copy=False).copy()
        if tree_segment_ids.size > 0:
            tree_mask = np.isin(segment, tree_segment_ids)
            instance[~tree_mask] = ignore_index
        instance[instance <= 0] = ignore_index

        coord_mm[cursor:end] = coord
        segment_mm[cursor:end] = segment
        instance_mm[cursor:end] = instance
        cursor = end

    if cursor != vertex_count:
        raise RuntimeError(
            f"Point count mismatch for {point_path}. header={vertex_count}, read={cursor}"
        )

    del coord_mm
    del segment_mm
    del instance_mm

    return {"scene": scene_name, "split": split, "points": vertex_count, "skipped": False}


def main():
    parser = argparse.ArgumentParser("Preprocess FOR-InstanceV2 into Pointcept format")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="FOR-InstanceV2 raw root, e.g. /data/tpy/raw-datasets/FOR-InstanceV2",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output root, e.g. data/for_instancev2",
    )
    parser.add_argument(
        "--semantic_map",
        type=str,
        default="1:0,2:1,3:2",
        help="Raw semantic to train id mapping, e.g. 1:0,2:1,3:2",
    )
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=-1,
        help="Ignore label value used by Pointcept",
    )
    parser.add_argument(
        "--tree_segment_ids",
        type=str,
        default="1,2",
        help="Segment ids considered as tree points for valid instance labels",
    )
    parser.add_argument(
        "--chunk_points",
        type=int,
        default=2_000_000,
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
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    semantic_map = parse_semantic_map(args.semantic_map)
    tree_segment_ids = parse_int_list(args.tree_segment_ids)

    tasks = build_tasks(dataset_root)
    if len(tasks) == 0:
        raise RuntimeError(f"No PLY files found under {dataset_root}")

    if args.max_scenes > 0:
        tasks = tasks[: args.max_scenes]

    print(f"[FOR-InstanceV2] Total scenes: {len(tasks)}")
    print(f"[FOR-InstanceV2] Semantic map: {semantic_map}")

    worker_kwargs = dict(
        output_root=str(output_root),
        semantic_map=semantic_map,
        ignore_index=int(args.ignore_index),
        tree_segment_ids=tree_segment_ids,
        chunk_points=int(args.chunk_points),
        overwrite=bool(args.overwrite),
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
    skipped = 0
    for r in results:
        split_counter[r["split"]] += 1
        if r["points"] is not None:
            point_counter[r["split"]] += int(r["points"])
        if r.get("skipped", False):
            skipped += 1

    meta = {
        "dataset": "FOR-InstanceV2",
        "ignore_index": int(args.ignore_index),
        "semantic_map_raw_to_trainid": semantic_map,
        "tree_segment_ids": tree_segment_ids.tolist(),
        "class_names": ["ground", "wood", "leaf"],
        "split_scene_count": split_counter,
        "split_point_count": point_counter,
        "skipped_scene_count": skipped,
        "chunk_points": int(args.chunk_points),
    }
    with (output_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[FOR-InstanceV2] Done")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
