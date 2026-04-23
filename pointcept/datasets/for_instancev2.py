"""
FOR-InstanceV2 dataset

Author: Open-source contributors
Please cite the original FOR-InstanceV2/ForestFormer3D papers if the code is helpful to you.
"""

import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class FORInstanceV2Dataset(DefaultDataset):
    """FOR-InstanceV2 数据集适配。

    该类默认读取 Pointcept 预处理后的目录结构：
    - coord.npy
    - segment.npy
    - instance.npy
    - (optional) strength.npy
    """

    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "strength",
        "segment",
        "instance",
    ]

    def __init__(self, tree_segment_ids=(1, 2), **kwargs):
        # 语义中代表“树”的类别 id（默认 wood/leaf）。
        self.tree_segment_ids = np.array(tree_segment_ids, dtype=np.int32)
        super().__init__(**kwargs)

    def get_data(self, idx):
        data_dict = super().get_data(idx)

        if "strength" in data_dict:
            data_dict["strength"] = data_dict["strength"].astype(np.float32)

        # Ground / ignore 点不参与实例分割，避免后续聚类时引入噪声实例。
        if data_dict["instance"].shape[0] == data_dict["segment"].shape[0]:
            if self.tree_segment_ids.size > 0:
                tree_mask = np.isin(data_dict["segment"], self.tree_segment_ids)
                data_dict["instance"][~tree_mask] = self.ignore_index
            else:
                data_dict["instance"][
                    data_dict["segment"] == self.ignore_index
                ] = self.ignore_index

            # 按 FOR-InstanceV2 的标注习惯，<=0 视为无效实例。
            data_dict["instance"][data_dict["instance"] <= 0] = self.ignore_index

        return data_dict
