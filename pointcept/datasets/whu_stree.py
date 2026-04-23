"""
WHU-Stree dataset

Author: Open-source contributors
Please cite the original WHU-Stree paper if the code is helpful to you.
"""

import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class WHUStreeDataset(DefaultDataset):
    """WHU-Stree 数据集适配。

    约定输入为 Pointcept 预处理后的标准目录：
    - coord.npy
    - strength.npy (由 intensity 归一化得到)
    - segment.npy
    - instance.npy
    """

    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "strength",
        "segment",
        "instance",
    ]

    def __init__(self, strict_instance=True, **kwargs):
        self.strict_instance = strict_instance
        super().__init__(**kwargs)

    def get_data(self, idx):
        data_dict = super().get_data(idx)

        if "strength" in data_dict:
            strength = data_dict["strength"].astype(np.float32)
            if strength.ndim == 1:
                strength = strength[:, None]
            data_dict["strength"] = strength

        # 与 segment 对齐实例标签，确保 ignored 点不会进入实例评估。
        if self.strict_instance and data_dict["instance"].shape[0] == data_dict["segment"].shape[0]:
            data_dict["instance"][data_dict["segment"] == self.ignore_index] = (
                self.ignore_index
            )
            data_dict["instance"][data_dict["instance"] <= 0] = self.ignore_index

        return data_dict
