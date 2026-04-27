"""
Single-image inference script for RF-DETR segmentation.
Loads the best checkpoint from training and visualizes predictions.
"""

import argparse
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


def predict_and_visualize(
    model,
    image_path: str,
    output_path: str,
    confidence_threshold: float = 0.5,
    dpi: int = 150,
):
    """Run inference on a single image and save the annotated result."""
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Run inference
    detections = model.predict(image, threshold=confidence_threshold)

    # Build labels
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(
            detections.data["class_name"], detections.confidence
        )
    ]

    # Annotate
    annotated = annotate(image_np, detections, labels=labels)

    # Save
    plot_images([annotated], output_path=output_path, dpi=dpi, cols=1)
    print(f"Saved inference result to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RF-DETR Segmentation Inference")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/inference_result.jpg",
        help="Path to save annotated output image.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint. If not set, uses config default.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Confidence threshold. Overrides config if set.",
    )
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    cleanup_gpu_memory()

    model_name = cfg.get("model_name", "RFDETRSegNano")
    output_dir = resolve_output_dir(cfg, clear=False)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        # Try multiple checkpoint names in order of preference
        for ckpt_name in ["checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"]:
            candidate = os.path.join(output_dir, ckpt_name)
            if os.path.isfile(candidate):
                checkpoint_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {output_dir}. "
                f"Please train the model first or provide a valid --checkpoint path."
            )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please train the model first or provide a valid --checkpoint path."
        )

    confidence_threshold = (
        args.threshold
        if args.threshold is not None
        else cfg.get("confidence_threshold", 0.5)
    )

    print(f"Loading model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    model = get_model(model_name, checkpoint_path)

    predict_and_visualize(
        model,
        image_path=args.image,
        output_path=args.output,
        confidence_threshold=confidence_threshold,
        dpi=cfg.get("dpi", 150),
    )


if __name__ == "__main__":
    main()
