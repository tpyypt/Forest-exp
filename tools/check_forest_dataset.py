"""
Forest dataset integrity checker for Pointcept preprocessed folders.

Supports:
- FOR-InstanceV2
- WHU-Stree
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def infer_dataset_type(dataset_root: Path) -> str:
    meta_path = dataset_root / "meta.json"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            dataset_name = str(meta.get("dataset", "")).lower()
            if "for-instancev2" in dataset_name:
                return "for_instancev2"
            if "whu-stree" in dataset_name:
                return "whu_stree"
        except Exception:
            pass

    name = dataset_root.name.lower()
    if "for" in name and "instance" in name:
        return "for_instancev2"
    if "whu" in name or "stree" in name:
        return "whu_stree"
    return "unknown"


def required_assets(dataset_type: str):
    if dataset_type == "for_instancev2":
        return ["coord", "segment", "instance"]
    if dataset_type == "whu_stree":
        return ["coord", "strength", "segment", "instance"]
    return ["coord", "segment", "instance"]


def load_array(path: Path):
    return np.load(path, mmap_mode="r")


def check_scene(scene_dir: Path, req_assets):
    errors = []
    arrays = {}

    for asset in req_assets:
        f = scene_dir / f"{asset}.npy"
        if not f.exists():
            errors.append(f"missing {asset}.npy")
            continue
        try:
            arrays[asset] = load_array(f)
        except Exception as e:
            errors.append(f"failed to load {asset}.npy: {e}")

    if "coord" in arrays:
        coord = arrays["coord"]
        if coord.ndim != 2 or coord.shape[1] != 3:
            errors.append(f"coord shape should be [N,3], got {tuple(coord.shape)}")
        n = coord.shape[0] if coord.ndim >= 1 else None
    else:
        n = None

    for key, arr in arrays.items():
        if key == "coord":
            continue
        if n is None:
            break
        if arr.ndim == 1:
            if arr.shape[0] != n:
                errors.append(f"{key} length mismatch: {arr.shape[0]} vs coord {n}")
        elif arr.ndim == 2:
            if arr.shape[0] != n:
                errors.append(f"{key} length mismatch: {arr.shape[0]} vs coord {n}")
        else:
            errors.append(f"{key} ndim should be 1 or 2, got {arr.ndim}")

    return errors, n


def check_split(dataset_root: Path, split: str, req_assets):
    split_dir = dataset_root / split
    if not split_dir.exists():
        return {
            "split": split,
            "exists": False,
            "scene_count": 0,
            "ok_scene_count": 0,
            "error_scene_count": 0,
            "point_count": 0,
            "errors": [f"split directory not found: {split_dir}"],
        }

    scenes = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    split_errors = []
    ok_scene_count = 0
    point_count = 0

    for scene in scenes:
        errors, n = check_scene(scene, req_assets)
        if errors:
            split_errors.append(f"{scene.name}: " + "; ".join(errors))
        else:
            ok_scene_count += 1
            point_count += int(n)

    return {
        "split": split,
        "exists": True,
        "scene_count": len(scenes),
        "ok_scene_count": ok_scene_count,
        "error_scene_count": len(scenes) - ok_scene_count,
        "point_count": point_count,
        "errors": split_errors,
    }


def main():
    parser = argparse.ArgumentParser("Check preprocessed FOR-InstanceV2 / WHU-Stree folders")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="auto",
        choices=["auto", "for_instancev2", "whu_stree"],
        help="Dataset type. auto tries meta.json and folder name inference.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated splits, e.g. train,val,test",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero code if any split/scene check fails",
    )
    parser.add_argument(
        "--max_errors_per_split",
        type=int,
        default=20,
        help="Maximum error lines to print for each split",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

    if args.dataset_type == "auto":
        dataset_type = infer_dataset_type(dataset_root)
    else:
        dataset_type = args.dataset_type

    req_assets = required_assets(dataset_type)
    split_list = [s.strip() for s in args.splits.split(",") if s.strip()]

    print(f"[check] dataset_root: {dataset_root}")
    print(f"[check] dataset_type: {dataset_type}")
    print(f"[check] required_assets: {req_assets}")
    print(f"[check] splits: {split_list}")

    any_error = False
    for split in split_list:
        stats = check_split(dataset_root, split, req_assets)
        print(
            f"[split={split}] scenes={stats['scene_count']} ok={stats['ok_scene_count']} "
            f"error={stats['error_scene_count']} points={stats['point_count']}"
        )

        if stats["errors"]:
            any_error = True
            print(f"[split={split}] errors:")
            for line in stats["errors"][: args.max_errors_per_split]:
                print("  -", line)
            if len(stats["errors"]) > args.max_errors_per_split:
                remain = len(stats["errors"]) - args.max_errors_per_split
                print(f"  ... ({remain} more errors omitted)")

    if any_error and args.strict:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
