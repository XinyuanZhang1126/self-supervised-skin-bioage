"""
Evaluation script for RF-DETR segmentation on the test set.
Loads the best checkpoint, runs inference, and visualizes results.
"""

import os

import numpy as np
from PIL import Image

import supervision as sv
from rfdetr import (
    RFDETRSegNano,
    RFDETRSegSmall,
    RFDETRSegMedium,
    RFDETRSegLarge,
    RFDETRSegXLarge,
    RFDETRSeg2XLarge,
)
from rfdetr.utilities.state_dict import clean_state_dict, _ckpt_args_get

from utils import (
    load_config,
    annotate,
    plot_images,
    cleanup_gpu_memory,
    resolve_dataset_path,
    resolve_output_dir,
)


def get_model(model_name: str, checkpoint_path: str):
    """Load a trained RF-DETR model from checkpoint."""
    import torch

    name_to_cls = {
        "RFDETRSegNano": RFDETRSegNano,
        "RFDETRSegSmall": RFDETRSegSmall,
        "RFDETRSegMedium": RFDETRSegMedium,
        "RFDETRSegLarge": RFDETRSegLarge,
        "RFDETRSegXLarge": RFDETRSegXLarge,
        "RFDETRSeg2XLarge": RFDETRSeg2XLarge,
    }
    cls_ = name_to_cls.get(model_name)
    if cls_ is None:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Initialize model (pretrain_weights loaded by default)
    model = cls_()

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    class_names = _ckpt_args_get(args, "class_names", None) or []
    num_classes = len(class_names) + 1  # +1 for background

    # Reinitialize detection head to match the checkpoint's class count
    if num_classes > 1:
        model.model.reinitialize_detection_head(num_classes)

    # Load checkpoint weights into the underlying nn.Module
    state_dict = clean_state_dict(ckpt["model"])
    model.model.model.load_state_dict(state_dict, strict=False)

    # Restore class_names from checkpoint args if available
    if class_names:
        model.model.class_names = class_names

    # Fix num_select mismatch (postprocess.num_select may exceed num_queries after reinit)
    num_select = getattr(model.model.postprocess, "num_select", None)
    num_queries = getattr(model.model.model, "num_queries", None)
    if num_select is not None and num_queries is not None and num_select > num_queries:
        model.model.postprocess.num_select = num_queries

    model.optimize_for_inference()
    return model


def main():
    cfg = load_config("config.yaml")
    cleanup_gpu_memory()

    model_name = cfg.get("model_name", "RFDETRSegNano")
    output_dir = resolve_output_dir(cfg, clear=False)

    # Try multiple checkpoint names in order of preference
    for ckpt_name in ["checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"]:
        checkpoint_path = os.path.join(output_dir, ckpt_name)
        if os.path.isfile(checkpoint_path):
            break
    else:
        raise FileNotFoundError(
            f"No checkpoint found in {output_dir}. "
            f"Please run training first."
        )

    # Resolve dataset path
    dataset_dir = resolve_dataset_path(cfg)
    test_dir = os.path.join(dataset_dir, "test")
    test_annotation_path = os.path.join(test_dir, "_annotations.coco.json")

    if not os.path.isfile(test_annotation_path):
        raise FileNotFoundError(
            f"Test annotation not found: {test_annotation_path}\n"
            f"Ensure the dataset has a test split with COCO annotations."
        )

    confidence_threshold = cfg.get("confidence_threshold", 0.5)
    plot_top_k = cfg.get("plot_top_k", 9)
    dpi = cfg.get("dpi", 150)

    print(f"Loading model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    model = get_model(model_name, checkpoint_path)

    # Load test dataset using supervision
    print(f"Loading test dataset from {test_dir} ...")
    test_dataset = sv.DetectionDataset.from_coco(
        images_directory_path=test_dir,
        annotations_path=test_annotation_path,
    )
    print(f"Test set size: {len(test_dataset)}")

    # Run inference on test images
    annotated_images = []
    titles = []

    for image_path, image, _ in test_dataset:
        if len(annotated_images) >= plot_top_k:
            break

        image_np = image
        detections = model.predict(image, threshold=confidence_threshold)

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(
                detections.data["class_name"], detections.confidence
            )
        ]

        annotated = annotate(image_np, detections, labels=labels)
        annotated_images.append(annotated)
        titles.append(os.path.basename(image_path))

    # Save visualization grid
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    viz_path = os.path.join(eval_dir, "test_predictions.jpg")

    if annotated_images:
        plot_images(
            annotated_images,
            titles=titles,
            output_path=viz_path,
            dpi=dpi,
            cols=3,
        )
    else:
        print("No images were annotated (test set may be empty).")

    print(f"\nEvaluation complete. Results saved to {eval_dir}")


if __name__ == "__main__":
    main()
