# RF-DETR Instance Segmentation – Local Fine-tuning (RTX 3090)

本项目将 RF-DETR 官方 Colab tutorial 转换为可在本地 NVIDIA RTX 3090（24GB VRAM）上直接运行的 Python 脚本。

## 项目结构

```
rf-detr-seg-local/
├── README.md                 # 本文档
├── requirements.txt          # Python 依赖
├── config.yaml               # 统一配置（模型、训练参数、数据路径、3090 优化）
├── utils.py                  # 工具函数：可视化、GPU 清理、配置加载
├── prepare_data.py           # 下载示例 COCO 数据集（零 Roboflow SDK 依赖）
├── train.py                  # 训练脚本
├── inference.py              # 单张图片推理 + 可视化
├── evaluate.py               # 在 test set 上评估 + 可视化结果网格
└── outputs/                  # 训练输出（checkpoints、日志、可视化图）
    └── checkpoint_best_total.pth
└── data/                     # 数据集根目录
    └── creacks/              # 示例数据集（auto-downloaded）
        ├── train/
        ├── valid/
        └── test/
```

## 快速开始

### 1. 安装环境

```bash
cd rf-detr-seg-local
pip install -r requirements.txt
```

### 2. 下载示例数据集（可选）

如果你暂时没有自己的 COCO 格式数据集，运行以下命令自动下载示例数据集：

```bash
python prepare_data.py
```

### 3. 修改配置

编辑 [`config.yaml`](config.yaml)：

- **使用自己的数据集**：设置 `use_custom_dataset: true`，并填写 `custom_dataset_path`。
- **使用示例数据集**：保持默认即可。

### 4. 训练

```bash
python train.py
```

训练完成后，最佳 checkpoint 保存在 `outputs/checkpoint_best_total.pth`。

### 5. 推理（单张图片）

```bash
python inference.py --image path/to/image.jpg --output result.jpg
```

### 6. 评估（Test Set）

```bash
python evaluate.py
```

评估结果和可视化图片保存在 `outputs/eval/`。

## 配置说明（config.yaml）

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `use_custom_dataset` | 是否使用自己的数据集 | `false` |
| `custom_dataset_path` | 自定义数据集根目录 | `""` |
| `model_name` | 模型名称：`RFDETRSegNano` / `RFDETRSegMedium` / `RFDETRSegLarge` | `RFDETRSegNano` |
| `epochs` | 训练轮数 | `5` |
| `batch_size` | 单卡 batch size（3090 优化为 8） | `8` |
| `grad_accum_steps` | 梯度累积步数（effective batch = 16） | `2` |
| `lr` | 学习率，`null` 表示使用模型默认 | `null` |
| `num_workers` | DataLoader worker 数 | `4` |
| `seed` | 随机种子 | `42` |
| `output_dir` | 训练输出目录 | `./outputs` |
| `dataset_root` | 数据集下载/存放目录 | `./data` |
| `confidence_threshold` | 推理置信度阈值 | `0.5` |
| `nms_threshold` | NMS 阈值 | `0.5` |
| `max_detections` | 最大检测数量 | `300` |
| `plot_top_k` | 可视化网格中图片数量 | `9` |

## 数据集格式要求

如果你使用自己的数据集，请确保目录结构如下（标准 COCO 格式）：

```
your-dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── img001.jpg
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   ├── img101.jpg
│   └── ...
└── test/
    ├── _annotations.coco.json
    ├── img201.jpg
    └── ...
```

- `train/`、`valid/`、`test/` 三个目录必须存在。
- 每个目录下必须有 `_annotations.coco.json`。
- 图片格式支持 `.jpg`、`.jpeg`、`.png`。

## 3090 显存优化建议

- 默认配置 `batch_size=8, grad_accum_steps=2` 在 RTX 3090（24GB）上运行 `RFDETRSegNano` 通常占用约 16-20GB VRAM，留有安全余量。
- 如果你使用更大的模型（如 `RFDETRSegMedium`），建议保持 `batch_size=4, grad_accum_steps=4` 或减小 `batch_size`。
- 如果出现 OOM，请在 [`config.yaml`](config.yaml) 中降低 `batch_size` 并相应提高 `grad_accum_steps`，保持 `batch_size × grad_accum_steps = 16`（或你期望的有效 batch size）。

## 各脚本设计细节

### `utils.py`

- `load_config(path)`：加载 YAML 配置并转换为 dict。
- `cleanup_gpu_memory()`：`gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.ipc_collect()`。
- `annotate(image, detections, labels, resolution_wh)`：使用 supervision 的 `MaskAnnotator` + `PolygonAnnotator` + `LabelAnnotator` 绘制分割掩码、多边形和标签。
- `plot_images(images, titles, output_path, dpi)`：将多张图片拼接为网格图并保存。

### `prepare_data.py`

- 纯 Python 实现，不依赖 `roboflow` SDK。
- 使用 `requests` + `tqdm` 下载 Roboflow 公开数据集的 zip 文件。
- 自动解压到 `./data/creacks/`。
- 如果目标目录已存在且非空，则跳过下载。

### `train.py`

1. 读取 [`config.yaml`](config.yaml)。
2. 根据配置初始化模型（`RFDETRSegNano()` 等）。
3. 确定数据集路径：
   - 若 `use_custom_dataset` 为 `true`，使用 `custom_dataset_path`。
   - 否则使用 `./data/creacks/`（示例数据集）。
4. 调用 `model.train(dataset_dir=..., epochs=..., batch_size=..., grad_accum_steps=...)` 开始训练。
5. 训练日志和 checkpoint 自动保存到 `output_dir`。

### `inference.py`

1. 读取配置，加载训练好的最佳 checkpoint。
2. 对输入图片进行推理，获取 `Detections`。
3. 调用 `annotate()` 绘制结果。
4. 保存可视化结果到指定路径。

### `evaluate.py`

1. 读取配置，加载最佳 checkpoint。
2. 使用 `supervision.DetectionDataset.from_coco()` 加载 test set。
3. 对 test set 进行 batch inference。
4. 使用 `annotate()` 绘制前 `plot_top_k` 张图片的推理结果。
5. 保存网格图到 `outputs/eval/test_predictions.jpg`。

## 与 Colab Notebook 的差异

| 项目 | Colab Notebook | 本项目 |
|------|----------------|--------|
| 环境管理 | `!pip install` in cell | `requirements.txt` + `pip install` |
| 数据下载 | `roboflow.download_dataset()` | `prepare_data.py`（requests + zip） |
| API Key | `google.colab.userdata` | 无需 API Key（公开数据集） |
| 配置 | 硬编码在 cell 中 | [`config.yaml`](config.yaml) 统一管理 |
| 路径 | `/content/` | 相对路径 `./` |
| 训练启动 | cell 直接运行 | `python train.py` |
| 部署 | `model.deploy_to_roboflow()` | 已移除（纯本地工作流） |
| Batch Size | `batch_size=4, grad_accum_steps=4`（T4 16GB） | `batch_size=8, grad_accum_steps=2`（3090 24GB） |
