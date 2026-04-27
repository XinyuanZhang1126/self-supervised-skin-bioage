"""
Training script for RF-DETR instance segmentation fine-tuning.
Optimized for NVIDIA RTX 3090 (24GB VRAM).

【本文件是做什么的？】
这是一个"训练脚本"。简单来说，它的作用就是：
1. 读取你写的配置文件（config.yaml）
2. 准备好训练所需的数据和模型
3. 启动训练过程，让AI模型学习如何识别图片中的目标并分割出它们的轮廓
4. 训练完成后，把学好的模型保存到硬盘上

【什么是RF-DETR？】
RF-DETR是一种先进的视觉模型（AI"眼睛"），它能同时做两件事：
- 检测（Detection）：在图片中找到目标物体，画个框框住它
- 分割（Segmentation）：不仅找到目标，还要精确地描出目标的轮廓（像素级）

【什么是Fine-tuning（微调）？】
想象模型是一个已经读过很多书的大学生（预训练），但它还不懂你特定领域的东西。
Fine-tuning就是让它再读一遍你提供的"专业教材"（你的数据集），
这样它就能专门擅长识别你关心的物体了。
"""

# =============================================================================
# 第一步：导入必要的工具库
# =============================================================================
# os: 操作系统接口，用来处理文件路径、创建文件夹等
import os
# random: 随机数生成器，用来设置随机种子，让实验结果可复现
import random

# numpy: Python最著名的科学计算库，处理数组和矩阵运算
import numpy as np
# torch: PyTorch，当前最流行的深度学习框架，用来在GPU上训练神经网络
import torch

# 从rfdetr库导入三种不同大小的RF-DETR分割模型
# Nano = 最小最快，适合普通显卡
# Medium = 中等大小，平衡速度和精度
# Large = 最大最准，但需要更好的显卡
from rfdetr import (
    RFDETRSegNano,
    RFDETRSegSmall,
    RFDETRSegMedium,
    RFDETRSegLarge,
    RFDETRSegXLarge,
    RFDETRSeg2XLarge,
)
# 从我们自己的utils.py导入一些辅助函数
from utils import load_config, resolve_dataset_path, cleanup_gpu_memory, resolve_output_dir


# =============================================================================
# 第二步：设置随机种子（让实验可复现）
# =============================================================================
def set_seed(seed: int = 42):
    """
    设置随机种子，确保每次运行代码时，随机结果都是一样的。

    【为什么要设置随机种子？】
    深度学习训练过程中有很多随机因素，比如：
    - 模型权重的初始随机值
    - 数据加载时的随机打乱顺序
    - Dropout等层的随机失活

    如果不设种子，每次训练结果都会略有不同，这样就无法比较
    "改了参数A之后，效果真的变好了吗"。
    设置种子后，只要代码和数据不变，结果就一模一样，方便调试和对比。

    参数:
        seed: 随机种子数字。42是程序员界的一个梗（《银河系漫游指南》里宇宙终极答案），
              你可以改成任意整数，比如2024、12345等。
    """
    random.seed(seed)       # 设置Python内置随机库的随机种子
    np.random.seed(seed)    # 设置NumPy的随机种子
    torch.manual_seed(seed) # 设置PyTorch的随机种子（CPU上）
    if torch.cuda.is_available():
        # 如果电脑有NVIDIA显卡，也设置显卡上的随机种子
        # torch.cuda.is_available() 会检查是否有可用的GPU
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 第三步：根据名字创建模型
# =============================================================================
def get_model(model_name: str):
    """
    根据模型名称字符串，创建对应的RF-DETR分割模型实例。

    【什么是模型（Model）？】
    模型就是一个巨大的数学函数，有上亿甚至上百亿个参数（可调整的旋钮）。
    训练的过程就是不断调整这些旋钮，让模型的输出越来越接近正确答案。

    【三种模型有什么区别？】
    - RFDETRSegNano:   参数量最少，推理速度最快，适合显存小的显卡（如8GB）
    - RFDETRSegMedium: 参数量中等，速度和精度比较平衡，适合16GB显存
    - RFDETRSegLarge:  参数量最多，精度最高但最慢，适合24GB+显存（如RTX 3090/4090）

    预训练权重：模型在创建时会自动加载官方预训练好的权重文件（如 rf-detr-seg-nano.pt），
    这意味着模型一开始就已经"见过世面"了，不是从零开始学习的。

    参数:
        model_name: 模型名称字符串，必须在 {"RFDETRSegNano", "RFDETRSegMedium", "RFDETRSegLarge"} 中

    返回:
        创建好的模型实例
    """
    # 建立一个"名字 -> 模型类"的对应字典
    name_to_cls = {
        "RFDETRSegNano": RFDETRSegNano,
        "RFDETRSegSmall": RFDETRSegSmall,
        "RFDETRSegMedium": RFDETRSegMedium,
        "RFDETRSegLarge": RFDETRSegLarge,
        "RFDETRSegXLarge": RFDETRSegXLarge,
        "RFDETRSeg2XLarge": RFDETRSeg2XLarge,
    }
    # 从字典中查找对应的模型类
    cls_ = name_to_cls.get(model_name)
    if cls_ is None:
        # 如果用户写错了名字，就报错并提示可用的选项
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Choose from {list(name_to_cls.keys())}"
        )
    # 调用模型类来创建实例，括号 () 表示"实例化"
    return cls_()


# =============================================================================
# 第四步：主函数——整个训练流程的"总指挥"
# =============================================================================
def main():
    """
    主函数，按顺序执行训练前的所有准备工作，最后启动训练。

    【训练流程概览】
    1. 读取配置（config.yaml）→ 知道训练多久、用多大批次等
    2. 设置随机种子 → 保证可复现
    3. 检查数据集 → 确认图片和标注文件都在
    4. 创建输出目录 → 存放训练好的模型
    5. 初始化模型 → 加载预训练权重
    6. 清理GPU内存 → 避免上次运行的残留占用显存
    7. 读取训练超参数 → 学习率、批次大小等
    8. 启动训练 → 调用 model.train()
    """

    # -------------------------------------------------------------------------
    # 4.1 读取配置文件
    # -------------------------------------------------------------------------
    # config.yaml 是一个YAML格式的文本文件，里面存放了所有训练参数。
    # 把参数写在配置文件里，而不是直接写在代码里，是为了：
    # - 不用改代码就能调整训练设置
    # - 方便分享和记录实验配置
    cfg = load_config("config.yaml")

    # -------------------------------------------------------------------------
    # 4.2 设置随机种子
    # -------------------------------------------------------------------------
    # cfg.get("seed", 42) 的意思是：
    # "从配置里读取seed的值，如果没写，就用默认值42"
    set_seed(cfg.get("seed", 42))

    # -------------------------------------------------------------------------
    # 4.3 解析并验证数据集路径
    # -------------------------------------------------------------------------
    # 数据集目录里应该有三个子文件夹：train（训练集）、valid（验证集）、test（测试集）
    # resolve_dataset_path 会根据配置自动找到正确的路径
    dataset_dir = resolve_dataset_path(cfg)

    # os.path.isdir() 检查这个路径是否真的是一个存在的文件夹
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}\n"
            f"Please run 'python prepare_data.py' first, or set a valid custom_dataset_path."
        )

    # -------------------------------------------------------------------------
    # 4.4 检查数据集的三种划分是否都存在
    # -------------------------------------------------------------------------
    # 【什么是 train / valid / test？】
    # - train（训练集）：用来训练模型的数据，模型会反复看这些数据来学习
    # - valid（验证集）：用来在训练过程中评估模型表现，调整超参数，防止过拟合
    # - test（测试集）：训练完全结束后，用来最终评估模型真实水平的数据
    #
    # 三者必须严格分开！如果测试集被模型提前"偷看"过，评估结果就不准了。
    for split in ["train", "valid", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Missing required split directory: {split_dir}"
            )

    # -------------------------------------------------------------------------
    # 4.5 创建输出目录（保存模型和日志的地方）
    # -------------------------------------------------------------------------
    # resolve_output_dir 会按数据集名称在 outputs 下创建子文件夹，
    # 并且只在同一数据集重复运行时清除对应的子文件夹，不会影响其他数据集。
    output_dir = resolve_output_dir(cfg)

    # -------------------------------------------------------------------------
    # 4.6 初始化模型
    # -------------------------------------------------------------------------
    # 从配置中读取要使用的模型名称，默认用最小的 Nano 模型
    model_name = cfg.get("model_name", "RFDETRSegNano")
    print(f"Initializing model: {model_name}")
    # 调用上面定义的 get_model 函数来创建模型实例
    model = get_model(model_name)

    # -------------------------------------------------------------------------
    # 4.7 清理GPU内存
    # -------------------------------------------------------------------------
    # 【为什么要清理GPU内存？】
    # GPU显存是有限的资源（比如RTX 3090有24GB）。
    # 如果上一次训练没有正常结束，或者PyTorch缓存了一些内存没释放，
    # 可能导致本次训练一开始就没有足够的显存可用。
    # cleanup_gpu_memory() 会强制清理这些残留的缓存，确保"从零开始"。
    cleanup_gpu_memory()

    # -------------------------------------------------------------------------
    # 4.8 读取训练超参数（Hyperparameters）
    # -------------------------------------------------------------------------
    # 【什么是超参数？】
    # 超参数是你在训练"之前"就设定好的参数，模型自己不会学它们，需要人来调整。
    # 常见的超参数包括：

    # epochs（轮数）：把整个训练集看多少遍。
    # 比如 epochs=5 表示模型会把所有训练图片看5遍。
    # 值太小 → 模型还没学会就停了（欠拟合）
    # 值太大 → 模型记住了训练集的细节，对新图片表现差（过拟合）
    epochs = cfg.get("epochs", 5)

    # batch_size（批次大小）：每次同时处理多少张图片。
    # GPU会一次性把batch_size张图片送入模型，计算平均损失后统一更新模型参数。
    # 值越大 → 训练越快，梯度估计越稳定，但占用的显存越多
    # 值越小 → 显存占用少，但训练速度慢，梯度波动大
    batch_size = cfg.get("batch_size", 8)

    # grad_accum_steps（梯度累积步数）：每多少个批次才更新一次模型参数。
    # 【为什么要梯度累积？】
    # 假设你想达到 "有效批次大小 = 16" 的效果，但你的显卡只能放下8张图片。
    # 那么你可以设置 batch_size=8, grad_accum_steps=2，
    # 这样模型会：
    #   第1步：处理8张图，计算损失，但不更新参数（把梯度存起来）
    #   第2步：再处理8张图，计算损失，把梯度加到上一次的上面
    #   最后：把累积的梯度除以2取平均，更新一次参数
    # 效果上等同于 batch_size=16，但显存只需要装8张图！
    grad_accum_steps = cfg.get("grad_accum_steps", 2)

    # lr（Learning Rate，学习率）：模型每次更新参数的"步子"有多大。
    # 学习率太大 → 步子迈太大，可能错过最优解，甚至发散（loss变成NaN）
    # 学习率太小 → 步子迈太小，训练慢得像蜗牛，可能陷入局部最优
    # None 表示使用模型默认的学习率
    lr = cfg.get("lr", None)

    # num_workers（数据加载的工作进程数）：用几个CPU进程来预加载图片数据。
    # 训练时GPU计算和数据加载是并行的：
    # - GPU在训练当前批次时
    # - CPU的几个worker在后台把下一批次的图片从硬盘读到内存，做好预处理
    # 这样GPU就不用等待数据，可以满负荷工作。
    # 一般设置为CPU核心数的一半左右，太大反而会因为进程间竞争而变慢。
    num_workers = cfg.get("num_workers", 4)

    # -------------------------------------------------------------------------
    # 4.9 打印训练配置（方便检查对不对）
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Training Configuration:")
    print(f"  Dataset:       {dataset_dir}")       # 数据集位置
    print(f"  Output dir:    {output_dir}")       # 模型保存位置
    print(f"  Model:         {model_name}")        # 使用的模型
    print(f"  Epochs:        {epochs}")            # 训练轮数
    print(f"  Batch size:    {batch_size}")        # 每批次图片数
    print(f"  Grad accum:    {grad_accum_steps}")  # 梯度累积步数
    print(f"  Effective BS:  {batch_size * grad_accum_steps}")  # 实际等效批次大小
    print(f"  LR:            {lr if lr else 'default'}")  # 学习率
    print(f"  Num workers:   {num_workers}")       # 数据加载进程数
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 4.10 组装训练参数，启动训练
    # -------------------------------------------------------------------------
    # 把上面的参数打包成一个字典（dict），传给模型的 train 方法
    train_kwargs = {
        "dataset_dir": dataset_dir,       # 数据集根目录
        "epochs": epochs,                 # 训练轮数
        "batch_size": batch_size,         # 批次大小
        "grad_accum_steps": grad_accum_steps,  # 梯度累积步数
        "num_workers": num_workers,       # 数据加载进程数
        "output_dir": output_dir,         # 输出目录
    }
    # 只有用户手动指定了学习率时，才把它加入参数列表
    # 如果用户没写（lr=None），就让模型使用自己默认的学习率策略
    if lr is not None:
        train_kwargs["lr"] = lr

    # 如果用户指定了输入分辨率，传给模型
    # 8000x4000 的原图太大会导致验证时显存溢出，降低分辨率可以避免 OOM
    resolution = cfg.get("resolution", None)
    if resolution is not None:
        train_kwargs["resolution"] = resolution

    # 【模型开始训练！】
    # model.train() 是 rfdetr 库封装好的训练函数，内部会：
    # 1. 读取 COCO 格式的标注文件（_annotations.coco.json）
    # 2. 构建数据加载器（DataLoader）
    # 3. 设置优化器（如AdamW）和学习率调度器
    # 4. 进入训练循环：前向传播 → 计算损失 → 反向传播 → 更新参数
    # 5. 定期在验证集上评估，保存表现最好的模型
    # 6. 训练结束后保存最终模型到 output_dir
    model.train(**train_kwargs)

    # -------------------------------------------------------------------------
    # 4.11 训练完成！
    # -------------------------------------------------------------------------
    print(f"\nTraining complete. Checkpoints saved to: {output_dir}")


# =============================================================================
# 程序入口
# =============================================================================
# 这行代码的意思是："只有当直接运行 train.py 时，才执行 main() 函数"
# 如果是从别的文件 import train，则不会自动运行 main()
# 这是Python的标准写法，避免模块被导入时意外执行代码
if __name__ == "__main__":
    main()
