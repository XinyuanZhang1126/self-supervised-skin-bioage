#!/usr/bin/env python3
"""
Resize all images in data/bloodshot_overfit/{train,valid,test}/
to a target max dimension while keeping aspect ratio, and update
their COCO annotation files accordingly.
"""

import json
import os
from pathlib import Path

from PIL import Image


def resize_dataset(base_dir: str, target_max_dim: int = 2000):
    splits = ["train", "valid", "test"]

    for split in splits:
        split_dir = Path(base_dir) / split
        if not split_dir.exists():
            print(f"Skipping missing split: {split_dir}")
            continue

        anno_path = split_dir / "_annotations.coco.json"
        if not anno_path.exists():
            print(f"Skipping missing annotation: {anno_path}")
            continue

        with open(anno_path, "r") as f:
            coco = json.load(f)

        # Map image_id -> original (width, height) before resizing
        orig_sizes = {}

        for img in coco.get("images", []):
            img_filename = img["file_name"]
            img_path = split_dir / img_filename

            if not img_path.exists():
                print(f"  Image not found, skipping: {img_path}")
                continue

            with Image.open(img_path) as pil_img:
                orig_w, orig_h = pil_img.size
                orig_sizes[img["id"]] = (orig_w, orig_h)

                scale = target_max_dim / max(orig_w, orig_h)
                new_w = int(round(orig_w * scale))
                new_h = int(round(orig_h * scale))

                resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                if resized.mode in ("RGBA", "P"):
                    resized = resized.convert("RGB")
                resized.save(img_path, format="JPEG", quality=95)

            # Update COCO image metadata
            img["width"] = new_w
            img["height"] = new_h

            print(f"  {split}/{img_filename}: {orig_w}x{orig_h} -> {new_w}x{new_h}")

        # Update annotations
        for ann in coco.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in orig_sizes:
                continue

            orig_w, orig_h = orig_sizes[img_id]
            scale = target_max_dim / max(orig_w, orig_h)
            scale_x = scale
            scale_y = scale

            # Update segmentation polygons
            segs = ann.get("segmentation", [])
            if isinstance(segs, list):
                new_segs = []
                for poly in segs:
                    if isinstance(poly, list):
                        new_poly = []
                        for i in range(0, len(poly), 2):
                            x = poly[i] * scale_x
                            y = poly[i + 1] * scale_y
                            new_poly.extend([x, y])
                        new_segs.append(new_poly)
                    else:
                        new_segs.append(poly)
                ann["segmentation"] = new_segs

            # Update bbox [x, y, width, height]
            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                ann["bbox"] = [
                    bbox[0] * scale_x,
                    bbox[1] * scale_y,
                    bbox[2] * scale_x,
                    bbox[3] * scale_y,
                ]

            # Update area
            if "area" in ann:
                ann["area"] = ann["area"] * (scale_x * scale_y)

        # Overwrite annotation file
        with open(anno_path, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"  Updated annotations: {anno_path}")


if __name__ == "__main__":
    DATASET_BASE = "data/bloodshot_overfit"
    resize_dataset(DATASET_BASE, target_max_dim=2000)
