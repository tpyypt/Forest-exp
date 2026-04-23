"""FOR-InstanceV2 数据配置基座（阶段 2：数据增强与数据构建）。

该文件仅定义 data builder 相关变量，不绑定具体模型。
后续模型配置可直接复用：
- for_instancev2_semseg_data
- for_instancev2_insseg_data
"""

for_instancev2_dataset_type = "FORInstanceV2Dataset"
for_instancev2_data_root = "data/for_instancev2"
for_instancev2_ignore_index = -1
for_instancev2_class_names = ["ground", "wood", "leaf"]
for_instancev2_num_classes = len(for_instancev2_class_names)

# 实例分割时忽略 ground（id=0）和 ignore（id=-1）。
for_instancev2_segment_ignore_index = (-1, 0)


# 语义分割 data builder -------------------------------------------------------
for_instancev2_semseg_data = dict(
    num_classes=for_instancev2_num_classes,
    ignore_index=for_instancev2_ignore_index,
    names=for_instancev2_class_names,
    train=dict(
        type=for_instancev2_dataset_type,
        split="train",
        data_root=for_instancev2_data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            # 森林点云稀疏区域较多，dropout 强度控制在较低范围避免欠采样。
            dict(
                type="RandomDropout",
                dropout_ratio=0.1,
                dropout_application_ratio=0.3,
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            # 允许轻微俯仰扰动，提高跨航线/跨设备鲁棒性。
            dict(type="RandomRotate", angle=[-1 / 128, 1 / 128], axis="x", p=0.3),
            dict(type="RandomRotate", angle=[-1 / 128, 1 / 128], axis="y", p=0.3),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.002, clip=0.01),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # OOM 防护：训练裁剪点数上限，避免单块密林样本显存峰值过高。
            dict(type="SphereCrop", point_max=220000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=for_instancev2_dataset_type,
        split="val",
        data_root=for_instancev2_data_root,
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
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=for_instancev2_dataset_type,
        split="test",
        data_root=for_instancev2_data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
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
                    feat_keys=("coord",),
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
for_instancev2_insseg_data = dict(
    num_classes=for_instancev2_num_classes,
    ignore_index=for_instancev2_ignore_index,
    names=for_instancev2_class_names,
    train=dict(
        type=for_instancev2_dataset_type,
        split="train",
        data_root=for_instancev2_data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout",
                dropout_ratio=0.1,
                dropout_application_ratio=0.3,
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 128, 1 / 128], axis="x", p=0.3),
            dict(type="RandomRotate", angle=[-1 / 128, 1 / 128], axis="y", p=0.3),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.002, clip=0.01),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # 与 semseg 一致的点数上限，降低 PointGroup 聚类时的显存压力。
            dict(type="SphereCrop", point_max=220000, mode="random"),
            dict(
                type="InstanceParser",
                segment_ignore_index=for_instancev2_segment_ignore_index,
                instance_ignore_index=for_instancev2_ignore_index,
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
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=for_instancev2_dataset_type,
        split="val",
        data_root=for_instancev2_data_root,
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
                segment_ignore_index=for_instancev2_segment_ignore_index,
                instance_ignore_index=for_instancev2_ignore_index,
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
                feat_keys=("coord",),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=for_instancev2_dataset_type,
        split="test",
        data_root=for_instancev2_data_root,
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
                segment_ignore_index=for_instancev2_segment_ignore_index,
                instance_ignore_index=for_instancev2_ignore_index,
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
                feat_keys=("coord",),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
)
