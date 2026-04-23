# FOR-InstanceV2 与 WHU-Stree 下游任务快速开始

本文档对应 Pointcept 内新增的两个单模态森林点云数据集适配：

- FOR-InstanceV2
- WHU-Stree

覆盖内容：

- 数据预处理
- 语义分割训练/测试
- 实例分割训练/测试


## 1. 数据预处理

### 1.1 FOR-InstanceV2

原始目录示例：

- `${RAW_FOR_ROOT}/train_val_data/*.ply`
- `${RAW_FOR_ROOT}/test_data/*.ply`

执行预处理：

```bash
python pointcept/datasets/preprocessing/for_instancev2/preprocess_for_instancev2.py \
  --dataset_root ${RAW_FOR_ROOT} \
  --output_root data/for_instancev2 \
  --semantic_map 1:0,2:1,3:2 \
  --tree_segment_ids 1,2 \
  --chunk_points 2000000 \
  --num_workers 1
```

等价 shell 包装脚本：

```bash
bash pointcept/datasets/preprocessing/for_instancev2/preprocess_for_instancev2.sh \
  -d ${RAW_FOR_ROOT} \
  -o data/for_instancev2
```

说明：

- 若内存紧张，保持 `--num_workers 1` 并适当减小 `--chunk_points`。
- 输出目录会生成 `meta.json`，记录 split 点数统计和映射信息。


### 1.2 WHU-Stree

支持两种原始组织方式：

- `split_samples/{train,val,test}/...`
- `road_id/PCD/*.ply + recommended_*_split.txt`

执行预处理：

```bash
python pointcept/datasets/preprocessing/whu_stree/preprocess_whu_stree.py \
  --dataset_root ${RAW_WHU_ROOT} \
  --output_root data/whu_stree \
  --chunk_points 1000000 \
  --num_workers 1 \
  --test_label_source reference \
  --intensity_scale 65535
```

等价 shell 包装脚本：

```bash
bash pointcept/datasets/preprocessing/whu_stree/preprocess_whu_stree.sh \
  -d ${RAW_WHU_ROOT} \
  -o data/whu_stree
```

说明：

- 默认会把稀疏物种标签重映射为连续 train id。
- 若希望保留原始稀疏标签，增加 `--keep_sparse_label`。


## 2. 训练配置

### 2.1 FOR-InstanceV2

- 语义分割：`configs/for_instancev2/semseg-pt-v3m1-0-base.py`
- 实例分割：`configs/for_instancev2/insseg-pointgroup-v1m2-0-ptv3-base.py`

### 2.2 WHU-Stree

- 语义分割：`configs/whu_stree/semseg-pt-v3m1-0-base.py`
- 实例分割：`configs/whu_stree/insseg-pointgroup-v1m2-0-ptv3-base.py`


## 2.5 数据完整性检查（推荐）

在正式训练前，建议先检查预处理产物：

```bash
python tools/check_forest_dataset.py --dataset_root data/for_instancev2 --strict
python tools/check_forest_dataset.py --dataset_root data/whu_stree --strict
```


## 3. 启动训练

以下命令以单机多卡脚本为例（按需修改 `-g`）。


### 3.1 FOR-InstanceV2 语义分割

```bash
bash scripts/train.sh \
  -d for_instancev2 \
  -c semseg-pt-v3m1-0-base \
  -n semseg-ptv3-base \
  -g 1
```


### 3.2 FOR-InstanceV2 实例分割

```bash
bash scripts/train.sh \
  -d for_instancev2 \
  -c insseg-pointgroup-v1m2-0-ptv3-base \
  -n insseg-pg-ptv3-base \
  -g 1
```


### 3.3 WHU-Stree 语义分割

```bash
bash scripts/train.sh \
  -d whu_stree \
  -c semseg-pt-v3m1-0-base \
  -n semseg-ptv3-base \
  -g 1
```


### 3.4 WHU-Stree 实例分割

```bash
bash scripts/train.sh \
  -d whu_stree \
  -c insseg-pointgroup-v1m2-0-ptv3-base \
  -n insseg-pg-ptv3-base \
  -g 1
```


## 4. 启动测试

测试命令会默认读取 `exp/<dataset>/<exp_name>/config.py` 和模型权重。

语义分割示例：

```bash
bash scripts/test.sh \
  -d for_instancev2 \
  -n semseg-ptv3-base \
  -w model_best \
  -g 1
```

实例分割示例：

```bash
bash scripts/test.sh \
  -d whu_stree \
  -n insseg-pg-ptv3-base \
  -w model_best \
  -g 1
```


## 5. 注意事项

- FOR-InstanceV2 实例分割默认忽略 `ground`（segment id=0）。
- WHU-Stree 配置会优先读取 `data/whu_stree/meta.json` 的 `train_class_count` 自动设置类别数。
- 若你只做快速联调，可先在预处理脚本使用 `--max_scenes` 取小子集验证训练链路。