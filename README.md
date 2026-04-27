# 自监督皮肤生物年龄估计与衰老轨迹研究

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5.1-red)](https://pytorch.org/)
[![RF-DETR](https://img.shields.io/badge/RF--DETR-%E2%89%A51.4.0-green)](https://github.com/roboflow/rf-detr)

基于 **RF-DETR** 实例分割模型的本地微调框架，针对皮肤图像的多目标实例分割任务进行优化。支持 NVIDIA RTX 3090（24GB VRAM）等本地 GPU 环境，提供从数据准备、模型训练、推理到评估的完整工作流。

> **RF-DETR** 是 Roboflow 开源的实时目标检测与实例分割模型，具有高精度、低延迟的特点，支持 Nano / Small / Medium / Large / XLarge / 2XLarge 六种规模。

---

## 📁 项目结构

```
self-supervised-skin-bioage/
├── README.md                          # 项目文档
├── requirements.txt                   # Python 依赖
├── config.yaml                        # 统一配置文件（数据集、模型、训练参数）
│
├── train.py                           # 🏋️ 训练脚本：读取配置 → 加载数据 → 微调模型
├── inference.py                       # 🔍 单图推理：加载 checkpoint → 预测 → 可视化
├── evaluate.py                        # 📊 测试集评估：批量推理 → 结果网格可视化
├── plot_metrics.py                    # 📈 指标绘图：绘制 loss / mAP / mAR 曲线
│
├── utils.py                           # 🛠️ 工具函数：配置加载、GPU清理、可视化
├── prepare_data.py                    # ⬇️ 数据集下载：支持 Roboflow SDK 或纯 requests
├── resize_dataset.py                  # 📐 数据集缩放：等比缩放图片并同步更新 COCO 标注
├── fix_annotations.py                 # 🔧 标注修复：批量缩放 COCO 标注中的分割/边界框
│
└── how-to-finetune-rf-detr-on-segmentation-dataset.ipynb  # 📓 Colab 教程参考
```

---

## 🚀 快速开始

### 1. 环境安装

```bash
# 推荐使用 conda 安装 PyTorch (CUDA 12.1)
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装项目依赖
pip install -r requirements.txt
```

**环境要求：**
- NVIDIA Driver >= 525.60.13 (CUDA 12.x)
- NVIDIA GPU 推荐 >= 16GB VRAM（RTX 3090 24GB 已验证）
- Python >= 3.10

### 2. 准备数据集

#### 方式 A：使用自己的 COCO 格式数据集

编辑 [`config.yaml`](config.yaml)：

```yaml
use_custom_dataset: true
custom_dataset_path: "./data/your-dataset"
```

数据集目录结构：

```
your-dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── img001.jpg
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   └── ...
└── test/
    ├── _annotations.coco.json
    └── ...
```

#### 方式 B：下载示例数据集

```bash
python prepare_data.py
```

自动下载 Roboflow 公开数据集（`creacks` 裂缝检测数据集）到 `./data/creacks/`。

### 3. 训练模型

```bash
python train.py
```

训练完成后，最佳 checkpoint 保存在：
```
outputs/<dataset_name>/checkpoint_best_total.pth
```

支持三种最佳模型保存策略：
- `checkpoint_best_total.pth` — 综合最佳
- `checkpoint_best_ema.pth` — EMA 模型最佳
- `checkpoint_best_regular.pth` — 常规模型最佳

### 4. 单图推理

```bash
python inference.py --image path/to/image.jpg --output result.jpg
```

可选参数：
- `--checkpoint`：指定 checkpoint 路径（默认自动查找最佳模型）
- `--threshold`：置信度阈值（覆盖 config.yaml 配置）

### 5. 测试集评估

```bash
python evaluate.py
```

评估结果可视化网格保存在：
```
outputs/<dataset_name>/eval/test_predictions.jpg
```

### 6. 指标可视化

```bash
python plot_metrics.py
```

绘制训练过程中的 Loss、mAP@0.50、mAP@0.50:0.95、mAR 曲线，保存为：
```
outputs/<dataset_name>/metrics_plots.png
```

---

## ⚙️ 配置说明（config.yaml）

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `use_custom_dataset` | 是否使用自己的数据集 | `true` |
| `custom_dataset_path` | 自定义数据集根目录 | `"./data/creacks"` |
| `roboflow_api_key` | Roboflow API Key（下载私有数据集时需要） | `""` |
| `model_name` | 模型规模：`RFDETRSegNano` / `RFDETRSegSmall` / `RFDETRSegMedium` / `RFDETRSegLarge` / `RFDETRSegXLarge` / `RFDETRSeg2XLarge` | `RFDETRSegNano` |
| `epochs` | 训练轮数 | `5` |
| `batch_size` | 单卡 batch size（RTX 3090 推荐 8） | `8` |
| `grad_accum_steps` | 梯度累积步数（有效 batch = 16） | `2` |
| `lr` | 学习率，`null` 表示使用模型默认 | `null` |
| `num_workers` | DataLoader 工作进程数 | `4` |
| `resolution` | 模型输入分辨率（需能被 12 整除） | `312` |
| `seed` | 随机种子 | `42` |
| `output_dir` | 训练输出目录 | `./outputs` |
| `dataset_root` | 数据集根目录 | `./data` |
| `confidence_threshold` | 推理置信度阈值 | `0.5` |
| `plot_top_k` | 评估可视化网格中图片数量 | `9` |
| `dpi` | 可视化图像分辨率 | `150` |

---

## 🎨 模型规模选择

| 模型 | 参数量 | 速度 | 显存需求 | 适用场景 |
|------|--------|------|----------|----------|
| `RFDETRSegNano` | 最小 | 最快 | ~8GB | 快速原型、边缘设备 |
| `RFDETRSegSmall` | 较小 | 快 | ~10GB | 平衡速度与精度 |
| `RFDETRSegMedium` | 中等 | 中等 | ~16GB | 精度优先 |
| `RFDETRSegLarge` | 较大 | 慢 | ~20GB+ | 高精度需求 |
| `RFDETRSegXLarge` | 大 | 较慢 | ~22GB+ | 极致精度 |
| `RFDETRSeg2XLarge` | 最大 | 最慢 | ~24GB+ | 研究实验 |

> 💡 **RTX 3090（24GB）推荐配置**：`RFDETRSegNano` + `batch_size=8, grad_accum_steps=2`

---

## 💾 显存优化建议

- 默认配置 `batch_size=8, grad_accum_steps=2` 在 RTX 3090 上运行 `RFDETRSegNano` 占用约 16-20GB VRAM
- 使用更大模型时，建议降低 `batch_size` 并提高 `grad_accum_steps`，保持 `batch_size × grad_accum_steps = 16`
- 出现 OOM 时，可进一步降低 `batch_size` 或减小 `resolution`（需被 12 整除）

---

## 📦 核心脚本说明

### `train.py`
1. 读取 [`config.yaml`](config.yaml) 配置
2. 设置随机种子保证实验可复现
3. 解析数据集路径并验证 `train/valid/test` 结构
4. 按数据集名称创建隔离的输出目录（`outputs/<dataset_name>/`）
5. 初始化模型并自动加载官方预训练权重
6. 调用 `model.train()` 启动训练，自动保存最佳 checkpoint

### `inference.py`
1. 从 checkpoint 恢复模型权重和类别信息
2. 自动处理检测头重初始化（适配自定义数据集类别数）
3. 对输入图片进行推理并可视化分割掩码、多边形和标签

### `evaluate.py`
1. 加载测试集 COCO 标注
2. 批量推理测试图片
3. 绘制前 `plot_top_k` 张图片的推理结果网格

### `utils.py`
- `load_config()` — YAML 配置加载
- `cleanup_gpu_memory()` — 清理 GPU 显存
- `annotate()` — 使用 supervision 绘制分割掩码、多边形和标签
- `plot_images()` — 图片网格拼接保存
- `resolve_dataset_path()` / `resolve_output_dir()` — 路径解析

### `prepare_data.py`
- 支持两种下载模式：Roboflow SDK（认证）或纯 requests（公开数据集）
- 自动解压并验证数据集结构
- 数据集已存在时自动跳过

### `resize_dataset.py`
- 等比缩放图片到目标最大边长
- 同步更新 COCO 标注中的 `segmentation`、`bbox`、`area`

### `fix_annotations.py`
- 批量修复 COCO 标注的缩放比例
- 适用于图片已缩放但标注未同步更新的场景

---

## 📈 训练指标说明

| 指标 | 说明 |
|------|------|
| `train/loss` | 训练损失 |
| `val/loss` | 验证损失 |
| `val/mAP_50` | 基础模型 mAP@IoU=0.50 |
| `val/ema_mAP_50` | EMA 模型 mAP@IoU=0.50 |
| `val/mAP_50_95` | 基础模型 mAP@IoU=0.50:0.95 |
| `val/ema_mAP_50_95` | EMA 模型 mAP@IoU=0.50:0.95 |
| `val/mAR` | 基础模型平均召回率 |
| `val/ema_mAR` | EMA 模型平均召回率 |

> **EMA**（Exponential Moving Average）：指数移动平均模型，通常比基础模型更稳定、泛化更好。

---

## 🔗 相关资源

- [RF-DETR 官方仓库](https://github.com/roboflow/rf-detr)
- [Roboflow Notebooks](https://github.com/roboflow-ai/notebooks)
- [supervision 可视化库](https://github.com/roboflow/supervision)

---

## 📄 许可证

本项目代码基于 MIT 许可证开源。RF-DETR 模型遵循其官方许可证。
