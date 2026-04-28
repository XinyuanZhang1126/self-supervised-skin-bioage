#!/usr/bin/env python3
"""One-off fix to scale bloodshot_overfit COCO annotations by 0.25.

Images were already resized from 8000x4000 -> 2000x1000, but the first
resize_dataset.py run left the annotations unscaled. This script corrects
segmentation, bbox, and area values.
"""

import json
from pathlib import Path


def fix_annotations(base_dir: str, scale: float = 0.25):
    splits = ["train", "valid", "test"]
    for split in splits:
        anno_path = Path(base_dir) / split / "_annotations.coco.json"
        if not anno_path.exists():
            continue

        with open(anno_path, "r") as f:
            coco = json.load(f)

        for ann in coco.get("annotations", []):
            # segmentation
            segs = ann.get("segmentation", [])
            if isinstance(segs, list):
                new_segs = []
                for poly in segs:
                    if isinstance(poly, list):
                        new_poly = [v * scale for v in poly]
                        new_segs.append(new_poly)
                    else:
                        new_segs.append(poly)
                ann["segmentation"] = new_segs

            # bbox
            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                ann["bbox"] = [v * scale for v in bbox]

            # area
            if "area" in ann:
                ann["area"] = ann["area"] * (scale * scale)

        with open(anno_path, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"Fixed {anno_path}")


if __name__ == "__main__":
    fix_annotations("data/bloodshot_overfit", scale=0.25)
