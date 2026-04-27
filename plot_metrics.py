import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot RF-DETR training metrics")
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Path to metrics.csv. If not set, uses <output_dir>/<dataset_name>/metrics.csv from config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the plot. If not set, saves to the same directory as metrics.csv.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    metrics_path = args.metrics
    if metrics_path is None:
        from utils import load_config, resolve_output_dir
        cfg = load_config("config.yaml")
        metrics_path = os.path.join(resolve_output_dir(cfg, clear=False), "metrics.csv")

    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    # 读取 metrics.csv
    df = pd.read_csv(metrics_path)

    def safe_float(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return np.nan

    # 所有需要绘制的列
    plot_cols = [
        "epoch", "train/loss", "val/loss",
        "val/mAP_50", "val/ema_mAP_50",
        "val/mAP_50_95", "val/ema_mAP_50_95",
        "val/mAR", "val/ema_mAR"
    ]
    for col in plot_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_float)

    # 按 epoch 聚合：每个 epoch 取该列的非空值
    # CSV 中同一 epoch 有验证行（无 train/loss）和训练行（无 val/*）
    # 所以用 first() 取非空值不够，需要自定义聚合：取第一个非 NaN
    def first_valid(series):
        vals = series.dropna()
        return vals.iloc[0] if len(vals) > 0 else np.nan

    df_epoch = df.groupby("epoch", sort=True).agg(first_valid).reset_index()

    epochs = df_epoch["epoch"].values
    train_loss = df_epoch["train/loss"].values
    val_loss = df_epoch["val/loss"].values
    map50_base = df_epoch["val/mAP_50"].values
    map50_ema = df_epoch["val/ema_mAP_50"].values
    map5095_base = df_epoch["val/mAP_50_95"].values
    map5095_ema = df_epoch["val/ema_mAP_50_95"].values
    mar_base = df_epoch["val/mAR"].values
    mar_ema = df_epoch["val/ema_mAR"].values

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Training and Validation Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, marker="o", linestyle="-", color="C0", label="Training Loss")
    ax.plot(epochs, val_loss, marker="o", linestyle="--", color="C1", label="Validation Loss")
    ax.set_title("Training and Validation Loss")
    ax.set_xlabel("Epoch Number")
    ax.set_ylabel("Loss Value")
    ax.legend()
    ax.grid(True)

    # 2. Average Precision @0.50
    ax = axes[0, 1]
    ax.plot(epochs, map50_base, marker="o", linestyle="-", color="C0", label="Base Model")
    ax.plot(epochs, map50_ema, marker="o", linestyle="--", color="C1", label="EMA Model")
    ax.set_title("Average Precision @0.50")
    ax.set_xlabel("Epoch Number")
    ax.set_ylabel("AP50")
    ax.legend()
    ax.grid(True)

    # 3. Average Precision @0.50:0.95
    ax = axes[1, 0]
    ax.plot(epochs, map5095_base, marker="o", linestyle="-", color="C0", label="Base Model")
    ax.plot(epochs, map5095_ema, marker="o", linestyle="--", color="C1", label="EMA Model")
    ax.set_title("Average Precision @0.50:0.95")
    ax.set_xlabel("Epoch Number")
    ax.set_ylabel("AP")
    ax.legend()
    ax.grid(True)

    # 4. Average Recall @0.50:0.95
    ax = axes[1, 1]
    ax.plot(epochs, mar_base, marker="o", linestyle="-", color="C0", label="Base Model")
    ax.plot(epochs, mar_ema, marker="o", linestyle="--", color="C1", label="EMA Model")
    ax.set_title("Average Recall @0.50:0.95")
    ax.set_xlabel("Epoch Number")
    ax.set_ylabel("AR")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    output_path = args.output or os.path.join(
        os.path.dirname(metrics_path), "metrics_plots.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
