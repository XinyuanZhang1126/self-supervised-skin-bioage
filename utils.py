"""
Utility functions for RF-DETR segmentation fine-tuning.
Includes visualization helpers, GPU memory cleanup, and config loading.
"""

import gc
import os
import shutil
from typing import List, Optional, Tuple

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from PIL import Image

import supervision as sv


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def cleanup_gpu_memory():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def annotate(
    image: np.ndarray,
    detections: sv.Detections,
    labels: Optional[List[str]] = None,
    resolution_wh: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Annotate an image with segmentation masks, polygons, and labels.

    Args:
        image: RGB image as numpy array (H, W, 3).
        detections: supervision Detections object (must include mask field).
        labels: List of label strings, one per detection.
        resolution_wh: Optional (width, height) for scaling annotations.

    Returns:
        Annotated image as numpy array.
    """
    mask_annotator = sv.MaskAnnotator()
    polygon_annotator = sv.PolygonAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = image.copy()

    # Draw masks and polygons if masks are available
    if detections.mask is not None:
        annotated_image = mask_annotator.annotate(
            scene=annotated_image, detections=detections
        )
        annotated_image = polygon_annotator.annotate(
            scene=annotated_image, detections=detections
        )

    # Draw labels if provided
    if labels is not None:
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )

    return annotated_image


def plot_images(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    output_path: str = "output.jpg",
    dpi: int = 150,
    cols: int = 3,
):
    """
    Plot a grid of images and save to file.

    Args:
        images: List of RGB numpy arrays.
        titles: Optional list of titles for each subplot.
        output_path: Path to save the resulting figure.
        dpi: Resolution for saved figure.
        cols: Number of columns in the grid.
    """
    n = len(images)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, img in enumerate(images):
        axes[i].imshow(img)
        axes[i].axis("off")
        if titles is not None and i < len(titles):
            axes[i].set_title(titles[i])

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


def get_dataset_name(cfg: dict) -> str:
    """
    Extract a short dataset name from configuration.

    Returns:
        Dataset name string (e.g. 'creacks', 'bloodshot_overfit').
    """
    if cfg.get("use_custom_dataset", False):
        path = cfg.get("custom_dataset_path", "")
        if path:
            return os.path.basename(os.path.normpath(path))
    example = cfg.get("example_dataset", {})
    return example.get("name", "creacks")


def resolve_dataset_path(cfg: dict) -> str:
    """
    Resolve the dataset directory path based on configuration.

    Returns:
        Absolute path to the dataset root directory.
    """
    if cfg.get("use_custom_dataset", False):
        path = cfg.get("custom_dataset_path", "")
        if not path:
            raise ValueError(
                "use_custom_dataset is true but custom_dataset_path is empty."
            )
        return os.path.abspath(path)

    # Example dataset path
    dataset_root = cfg.get("dataset_root", "./data")
    example = cfg.get("example_dataset", {})
    name = example.get("name", "creacks")
    return os.path.abspath(os.path.join(dataset_root, name))


def resolve_output_dir(cfg: dict, clear: bool = True) -> str:
    """
    Resolve the output directory for a given dataset.

    The output directory is structured as:
        <base_output_dir>/<dataset_name>/

    When ``clear=True`` and the same dataset is run again, only its
    dedicated folder is cleared so that outputs from other datasets
    are preserved.  Evaluation/inference scripts should set
    ``clear=False`` to avoid deleting training checkpoints.

    Args:
        cfg: Configuration dictionary.
        clear: If True, remove existing contents of the dataset folder.

    Returns:
        Absolute path to the dataset-specific output directory.
    """
    base_output_dir = os.path.abspath(cfg.get("output_dir", "./outputs"))
    dataset_name = get_dataset_name(cfg)
    dataset_output_dir = os.path.join(base_output_dir, dataset_name)

    if os.path.isdir(dataset_output_dir):
        if clear:
            print(f"[INFO] Clearing previous outputs for dataset '{dataset_name}' ...")
            for entry in os.listdir(dataset_output_dir):
                entry_path = os.path.join(dataset_output_dir, entry)
                try:
                    if os.path.isdir(entry_path):
                        shutil.rmtree(entry_path)
                    else:
                        os.remove(entry_path)
                except Exception as exc:
                    print(f"[WARNING] Could not remove {entry_path}: {exc}")
    else:
        os.makedirs(dataset_output_dir, exist_ok=True)

    return dataset_output_dir
