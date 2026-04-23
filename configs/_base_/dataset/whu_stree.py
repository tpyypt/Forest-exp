"""WHU-Stree 数据配置基座（阶段 2：数据增强与数据构建）。

该文件仅定义 data builder 相关变量，不绑定具体模型。
后续模型配置可直接复用：
- whu_stree_semseg_data
- whu_stree_insseg_data
"""

import json
import os


whu_stree_dataset_type = "WHUStreeDataset"
whu_stree_data_root = "data/whu_stree"
whu_stree_ignore_index = -1

# 论文 benchmark 的 19 类命名（18 主类 + Others），
# 若预处理生成的类别数与此不一致，会自动降级为 class_{i} 命名。
whu_stree_benchmark_names_19 = [
    "PA",
    "CC",
    "LI",
    "OF",
    "KP",
    "PC",
    "ZS",
    "CD",
    "GB",
    "AA",
    "PS",
    "LL",
    "MM",
    "MG",
    "MS",
    "SJ",
    "AM",
    "PF",
    "Others",
]


def _load_class_count_from_meta(data_root, default_count=19):
    meta_path = os.path.join(data_root, "meta.json")
    if not os.path.exists(meta_path):
        return default_count
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        count = int(meta.get("train_class_count", default_count))
        return max(count, 1)
    except Exception:
        return default_count


whu_stree_num_classes = _load_class_count_from_meta(whu_stree_data_root, default_count=19)
if whu_stree_num_classes == len(whu_stree_benchmark_names_19):
    whu_stree_class_names = whu_stree_benchmark_names_19
else:
    whu_stree_class_names = [f"class_{i}" for i in range(whu_stree_num_classes)]


# 实例分割时忽略 ignore 类（默认仅 -1）。
whu_stree_segment_ignore_index = (-1,)


# 语义分割 data builder -------------------------------------------------------
whu_stree_semseg_data = dict(
    num_classes=whu_stree_num_classes,
    ignore_index=whu_stree_ignore_index,
    names=whu_stree_class_names,
    train=dict(
        type=whu_stree_dataset_type,
        split="train",
        data_root=whu_stree_data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            # 城市街树场景中树冠交叠明显，保留更多点以稳定边界学习。
            dict(
                type="RandomDropout",
                dropout_ratio=0.08,
                dropout_application_ratio=0.25,
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            # 轻量 x/y 倾斜增强以适配不同采集姿态。
            dict(type="RandomRotate", angle=[-1 / 96, 1 / 96], axis="x", p=0.35),
            dict(type="RandomRotate", angle=[-1 / 96, 1 / 96], axis="y", p=0.35),
            dict(type="RandomScale", scale=[0.85, 1.15]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.001, clip=0.005),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # OOM 防护：街道级场景建议训练阶段限制点数上限。
            dict(type="SphereCrop", point_max=260000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=whu_stree_dataset_type,
        split="val",
        data_root=whu_stree_data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=whu_stree_dataset_type,
        split="test",
        data_root=whu_stree_data_root,
        transform=[dict(type="CenterShift", apply_z=True)],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1.0, 1.0])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.0, 1.0]), dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)


# 实例分割 data builder -------------------------------------------------------
whu_stree_insseg_data = dict(
    num_classes=whu_stree_num_classes,
    ignore_index=whu_stree_ignore_index,
    names=whu_stree_class_names,
    train=dict(
        type=whu_stree_dataset_type,
        split="train",
        data_root=whu_stree_data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout",
                dropout_ratio=0.08,
                dropout_application_ratio=0.25,
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 96, 1 / 96], axis="x", p=0.35),
            dict(type="RandomRotate", angle=[-1 / 96, 1 / 96], axis="y", p=0.35),
            dict(type="RandomScale", scale=[0.85, 1.15]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.001, clip=0.005),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=260000, mode="random"),
            dict(
                type="InstanceParser",
                segment_ignore_index=whu_stree_segment_ignore_index,
                instance_ignore_index=whu_stree_ignore_index,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "instance_centroid",
                    "bbox",
                ),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=whu_stree_dataset_type,
        split="val",
        data_root=whu_stree_data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(
                type="InstanceParser",
                segment_ignore_index=whu_stree_segment_ignore_index,
                instance_ignore_index=whu_stree_ignore_index,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                    "name",
                ),
                feat_keys=("coord", "strength"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=whu_stree_dataset_type,
        split="test",
        data_root=whu_stree_data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(
                type="InstanceParser",
                segment_ignore_index=whu_stree_segment_ignore_index,
                instance_ignore_index=whu_stree_ignore_index,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                    "name",
                ),
                feat_keys=("coord", "strength"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
)
