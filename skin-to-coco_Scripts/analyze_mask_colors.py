#!/usr/bin/env python3
"""
Analyze mask color layers and output per-value visualizations + connected-components stats.

For every unique non-zero pixel value in each mask image:
- Save an isolated binary mask image named by its pixel value
- Output connected-component stats (count, total area, largest area, concentration)
- Generate an overlay visualization on the main face image

Usage:
    python analyze_mask_colors.py
"""

import json
from pathlib import Path

import cv2
import numpy as np

PATIENT_DIR = Path("../data/33732019834458734")
MASK_DIR = PATIENT_DIR / "mask"
MAIN_IMAGE = PATIENT_DIR / "Rgb.jpg"
OUT_DIR = Path("../visualizations/analysis")
REPORT_JSON = Path("mask_color_stats.json")

CATEGORY_ORDER = [
    "Below_Eye_Wrinkles.jpg",
    "Between_Wrinkles.jpg",
    "Fish_tail_Wrinkles.jpg",
    "Forehead_Wrinkles.jpg",
    "Nasal_Wrinkles.jpg",
    "Nose_root_Wrinkles.jpg",
    "Acne_Face.jpg",
    "Bloodshot_Face.jpg",
    "Brown_Tag_Face.jpg",
    "Comedones_Face.jpg",
    "Dark_Face.jpg",
    "Pore_Face.jpg",
    "Rgb_Tag.jpg",
    "Senstive_Face.jpg",
    "T_Oil_Face.jpg",
    "U_Oil_Face.jpg",
    "Uv_Ex_Tag_Face.jpg",
    "Uv_P_Face.jpg",
]


def analyze_mask(mask_path: Path, main_img: np.ndarray, vis_dir: Path):
    """Analyze a single mask and return per-value stats."""
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read mask: {mask_path}")

    # Resize main image to match mask dimensions for overlay
    main_resized = cv2.resize(main_img, (img.shape[1], img.shape[0]))
    main_rgb = cv2.cvtColor(main_resized, cv2.COLOR_BGR2RGB)

    unique_vals = np.unique(img[img > 0])
    if len(unique_vals) == 0:
        return []

    rows = []
    for v in unique_vals:
        v_int = int(v)
        binary = (img == v_int).astype(np.uint8)

        # Connected components stats
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        total_pixels = int(np.sum(areas))
        max_area = int(np.max(areas)) if len(areas) > 0 else 0
        num_components = int(len(areas))
        concentration = max_area / total_pixels if total_pixels > 0 else 0.0

        rows.append({
            "val": v_int,
            "total_pixels": total_pixels,
            "num_components": num_components,
            "max_area": max_area,
            "concentration": round(concentration, 3),
        })

        # Save isolated binary mask (grayscale, 0 or 255)
        binary_255 = binary * 255
        mask_out = vis_dir / f"{mask_path.stem}_val{v_int}.png"
        cv2.imwrite(str(mask_out), binary_255)

        # Save overlay visualization on main face image
        overlay = main_rgb.copy()
        green = np.zeros_like(overlay)
        green[binary_255 > 0] = (0, 255, 0)
        blended = cv2.addWeighted(overlay, 0.6, green, 0.4, 0)
        contours, _ = cv2.findContours(binary_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1, (255, 0, 0), 2)
        vis_out = vis_dir / f"{mask_path.stem}_val{v_int}_overlay.png"
        cv2.imwrite(str(vis_out), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    main = cv2.imread(str(MAIN_IMAGE))
    if main is None:
        raise ValueError(f"Cannot read main image: {MAIN_IMAGE}")

    all_stats = {}

    for mask_name in CATEGORY_ORDER:
        mask_path = MASK_DIR / mask_name
        if not mask_path.exists():
            print(f"[MISSING] {mask_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Mask: {mask_name}")
        print(f"{'='*60}")

        rows = analyze_mask(mask_path, main, OUT_DIR)
        if not rows:
            print("  No non-zero pixels found.")
            continue

        # Sort by pixel count descending for display
        rows_sorted = sorted(rows, key=lambda x: x["total_pixels"], reverse=True)
        print(f"  {'val':>5} {'pixels':>10} {'components':>10} {'max_area':>10} {'conc':>7}")
        for r in rows_sorted:
            print(
                f"  {r['val']:>5} {r['total_pixels']:>10} {r['num_components']:>10} "
                f"{r['max_area']:>10} {r['concentration']:>7.3f}"
            )

        all_stats[mask_name] = rows

    # Write JSON stats report
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Stats saved to: {REPORT_JSON}")
    print(f"Visualizations saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
