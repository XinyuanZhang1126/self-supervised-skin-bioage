#!/usr/bin/env python3
"""
Convert single-patient skin dataset to RF-DETR-Seg COCO format.

Features:
- Resize main image (Rgb.jpg) to 768x768 for RFDETRSeg2XLarge.
- Convert 18 mask images to polygon segmentations using color clustering:
    * Most masks: 2 dominant colors (FOV + lesion).
    * Pore_Face.jpg / Comedones_Face.jpg: 3 dominant colors.
  FOV colors are ignored; only lesion colors become annotations.
- Lesion counts (connected components) are printed per mask.
- Generate train/valid/test splits with identical content (single patient).
- Output per-category and full-overlay visualizations.
"""

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PATIENT_ID = "33732019834458734"
PATIENT_DIR = Path("data") / PATIENT_ID
MASK_DIR = PATIENT_DIR / "mask"
MAIN_IMAGE = PATIENT_DIR / "Rgb.jpg"
OUT_DIR = Path("data/skin_single_patient")
TARGET_SIZE = (768, 768)
VIS_DIR = Path("visualizations")

# 18 categories: mask_filename -> category info
CATEGORY_MAP = {
    "Below_Eye_Wrinkles.jpg": {"id": 1, "name": "Below_Eye_Wrinkles"},
    "Between_Wrinkles.jpg": {"id": 2, "name": "Between_Wrinkles"},
    "Fish_tail_Wrinkles.jpg": {"id": 3, "name": "Fish_tail_Wrinkles"},
    "Forehead_Wrinkles.jpg": {"id": 4, "name": "Forehead_Wrinkles"},
    "Nasal_Wrinkles.jpg": {"id": 5, "name": "Nasal_Wrinkles"},
    "Nose_root_Wrinkles.jpg": {"id": 6, "name": "Nose_root_Wrinkles"},
    "Acne_Face.jpg": {"id": 7, "name": "Acne_Face"},
    "Bloodshot_Face.jpg": {"id": 8, "name": "Bloodshot_Face"},
    "Brown_Tag_Face.jpg": {"id": 9, "name": "Brown_Tag_Face"},
    "Comedones_Face.jpg": {"id": 10, "name": "Comedones_Face"},
    "Dark_Face.jpg": {"id": 11, "name": "Dark_Face"},
    "Pore_Face.jpg": {"id": 12, "name": "Pore_Face"},
    "Rgb_Tag.jpg": {"id": 13, "name": "Rgb_Tag"},
    "Senstive_Face.jpg": {"id": 14, "name": "Senstive_Face"},
    "T_Oil_Face.jpg": {"id": 15, "name": "T_Oil_Face"},
    "U_Oil_Face.jpg": {"id": 16, "name": "U_Oil_Face"},
    "Uv_Ex_Tag_Face.jpg": {"id": 17, "name": "Uv_Ex_Tag_Face"},
    "Uv_P_Face.jpg": {"id": 18, "name": "Uv_P_Face"},
}

# Masks that need 3 dominant colors instead of 2
THREE_COLOR_MASKS = {"Pore_Face.jpg", "Comedones_Face.jpg"}

# ---------------------------------------------------------------------------
# Manual color configuration (BGR tuples)
# If a mask is not listed here, auto-inference is used:
#   colors with few connected-components -> FOV (ignored)
#   colors with many connected-components -> lesion (annotated)
# ---------------------------------------------------------------------------
MASK_COLOR_CONFIG = {
    # Format example (update after manual inspection):
    # "Below_Eye_Wrinkles.jpg": {"fov": [(117, 74, 82)], "lesion": [(0, 254, 0)]},
}

# Minimum contour area (in resized pixels) to keep
MIN_CONTOUR_AREA = 0.0


def resize_main_image(src: Path, dst: Path, size: tuple):
    """Resize main image to target size with LANCZOS."""
    with Image.open(src) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        resized = img.resize(size, Image.Resampling.LANCZOS)
        resized.save(dst, format="JPEG", quality=95)
    print(f"Resized main image: {src} -> {dst} ({size[0]}x{size[1]})")


def _cluster_coarse_hsv(img: np.ndarray, n_clusters: int = 2, coarse_bins: int = 8):
    """
    Cluster mask pixels by coarse-quantized HSV colors.
    Returns (labels, centers_bgr, ys, xs) for foreground pixels.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    ys, xs = np.where(mask)
    pixels = hsv[ys, xs]

    if len(pixels) == 0:
        return np.array([]), [], np.array([]), np.array([])

    # Coarse quantize HSV
    h_bins = coarse_bins
    s_bins = coarse_bins
    v_bins = coarse_bins
    h_quant = (pixels[:, 0] // (180 // h_bins)).astype(np.uint8)
    s_quant = (pixels[:, 1] // (256 // s_bins)).astype(np.uint8)
    v_quant = (pixels[:, 2] // (256 // v_bins)).astype(np.uint8)
    quant = np.stack([h_quant, s_quant, v_quant], axis=1)

    uniques, counts = np.unique(quant, axis=0, return_counts=True)

    # Filter near-black (V < 2 bins or S < 1 bin)
    non_black = (uniques[:, 2] >= 2) & (uniques[:, 1] >= 1)
    uniques = uniques[non_black]
    counts = counts[non_black]

    if len(uniques) < n_clusters:
        return np.array([]), [], np.array([]), np.array([])

    # Coarse color centers (in HSV)
    h_centers = uniques[:, 0].astype(np.float32) * (180 // h_bins) + (180 // h_bins) // 2
    s_centers = uniques[:, 1].astype(np.float32) * (256 // s_bins) + (256 // s_bins) // 2
    v_centers = uniques[:, 2].astype(np.float32) * (256 // v_bins) + (256 // v_bins) // 2
    coarse_colors = np.stack([h_centers, s_centers, v_centers], axis=1)

    # Weighted sample for KMeans (sqrt count as weight)
    sample_weights = np.sqrt(counts)
    repeated = []
    for color, weight in zip(coarse_colors, sample_weights):
        n = max(1, int(weight))
        repeated.extend([color] * n)
    repeated = np.array(repeated)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(repeated)
    centers_hsv = kmeans.cluster_centers_.astype(np.float32)

    # Map each coarse unique to cluster
    coarse_labels = kmeans.predict(coarse_colors)
    quant_to_cluster = {tuple(q): lbl for q, lbl in zip(uniques, coarse_labels)}

    # Map each original pixel
    all_h = (hsv[:, :, 0] // (180 // h_bins)).astype(np.uint8)
    all_s = (hsv[:, :, 1] // (256 // s_bins)).astype(np.uint8)
    all_v = (hsv[:, :, 2] // (256 // v_bins)).astype(np.uint8)
    pixel_quants = np.stack([all_h[ys, xs], all_s[ys, xs], all_v[ys, xs]], axis=1)
    labels = np.array([quant_to_cluster.get(tuple(q), -1) for q in pixel_quants])

    # Convert HSV centers back to BGR for naming
    centers_bgr = []
    for c in centers_hsv:
        hsv_center = np.uint8([[c]])
        bgr_center = cv2.cvtColor(hsv_center, cv2.COLOR_HSV2BGR)[0, 0]
        centers_bgr.append(tuple(int(v) for v in bgr_center))

    return labels, centers_bgr, ys, xs


def extract_polygons_from_mask(mask_path: Path, target_wh: tuple, mask_name: str = ""):
    """
    Extract polygon segmentations from a color mask image.

    Strategy (coarse BGR quantization + KMeans):
    1. Load mask as BGR.
    2. Quantize BGR to coarse bins (8 per channel = 512 total).
    3. Find unique coarse colors, filter near-black.
    4. Weighted KMeans on coarse color centers (sqrt count as weight).
    5. Map ALL foreground pixels to their cluster.
    6. Determine FOV vs lesion by component count or MASK_COLOR_CONFIG.
    7. For 3-color masks, split lesion cluster with another K=2.
    8. Find contours -> approximate polygons -> scale to target_wh.

    Returns:
        (annotations, lesion_count)
    """
    img = cv2.imread(str(mask_path))
    if img is None:
        raise ValueError(f"Cannot read mask: {mask_path}")

    orig_h, orig_w = img.shape[:2]
    target_w, target_h = target_wh
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    # Step 1-5: coarse HSV quantization + KMeans K=2
    labels, centers_bgr, ys, xs = _cluster_coarse_hsv(img, n_clusters=2, coarse_bins=8)

    if len(labels) == 0:
        return [], 0

    # Build per-cluster binary masks and count components
    clusters = []
    for i in range(2):
        binary = np.zeros((orig_h, orig_w), dtype=np.uint8)
        binary[ys[labels == i], xs[labels == i]] = 255
        num_l, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        num_components = max(0, num_l - 1)
        median_bgr = tuple(int(v) for v in np.median(img[ys[labels == i], xs[labels == i]], axis=0)) if np.sum(labels == i) > 0 else centers_bgr[i]
        clusters.append({
            "label": i,
            "median_bgr": median_bgr,
            "binary": binary,
            "num_components": num_components,
            "total_pixels": int(np.sum(labels == i)),
        })

    # Sort by component count (fewer first)
    clusters.sort(key=lambda c: c["num_components"])

    # Determine FOV vs lesion
    config = MASK_COLOR_CONFIG.get(mask_name, {})
    if config:
        fov_colors = set(tuple(c) for c in config.get("fov", []))
        lesion_colors = set(tuple(c) for c in config.get("lesion", []))
        fov_clusters = []
        lesion_clusters = []
        for c in clusters:
            bgr = c["median_bgr"]
            dist_fov = min(np.linalg.norm(np.array(bgr) - np.array(fc)) for fc in fov_colors) if fov_colors else float('inf')
            dist_lesion = min(np.linalg.norm(np.array(bgr) - np.array(lc)) for lc in lesion_colors) if lesion_colors else float('inf')
            if dist_fov <= dist_lesion:
                fov_clusters.append(c)
            else:
                lesion_clusters.append(c)
    else:
        fov_clusters = [clusters[0]]
        lesion_clusters = [clusters[1]]

    # For 3-color masks, split lesion cluster further
    sub_masks = []
    sub_names = []

    for lc in lesion_clusters:
        if mask_name in THREE_COLOR_MASKS:
            lesion_ys = ys[labels == lc["label"]]
            lesion_xs = xs[labels == lc["label"]]
            if len(lesion_ys) == 0:
                continue
            # Create lesion-only image for sub-clustering
            lesion_img = np.zeros_like(img)
            lesion_img[lesion_ys, lesion_xs] = img[lesion_ys, lesion_xs]
            sub_labels, sub_centers, sub_ys, sub_xs = _cluster_coarse_hsv(lesion_img, n_clusters=2, coarse_bins=8)
            if len(sub_labels) == 0:
                continue
            for j in range(2):
                sub_m = np.zeros((orig_h, orig_w), dtype=np.uint8)
                sub_m[sub_ys[sub_labels == j], sub_xs[sub_labels == j]] = 255
                sub_masks.append(sub_m)
                sub_names.append(sub_centers[j])
        else:
            sub_masks.append(lc["binary"])
            sub_names.append(lc["median_bgr"])

    # Process each lesion sub-mask
    all_annotations = []
    total_lesion_cc = 0

    for sub_m, sub_name in zip(sub_masks, sub_names):
        num_l, _, _, _ = cv2.connectedComponentsWithStats(sub_m, connectivity=8)
        sub_cc = max(0, num_l - 1)
        total_lesion_cc += sub_cc

        contours, _ = cv2.findContours(sub_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sub_anns = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue

            epsilon = 0.005 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) < 3:
                continue

            poly = []
            for pt in approx:
                x = float(pt[0][0] * scale_x)
                y = float(pt[0][1] * scale_y)
                poly.extend([round(x, 2), round(y, 2)])

            x, y, w, h = cv2.boundingRect(approx)
            bbox = [
                round(float(x * scale_x), 2),
                round(float(y * scale_y), 2),
                round(float(w * scale_x), 2),
                round(float(h * scale_y), 2),
            ]

            scaled_area = float(area * scale_x * scale_y)

            sub_anns.append({
                "segmentation": [poly],
                "bbox": bbox,
                "area": round(scaled_area, 2),
            })

        all_annotations.extend(sub_anns)
        print(f"  {mask_path.name} {sub_name}: {sub_cc}cc, {len(sub_anns)} polygons")

    fov_info = ", ".join(f"{c['median_bgr']}={c['num_components']}cc" for c in fov_clusters)
    lesion_info = ", ".join(f"{n}" for n in sub_names)
    print(f"  {mask_path.name}: FOV=[{fov_info}], lesion=[{lesion_info}] -> {total_lesion_cc}cc, {len(all_annotations)} polygons")
    return all_annotations, total_lesion_cc


def build_coco_json(image_id: int, filename: str, width: int, height: int, all_annotations: list):
    """Build COCO-format JSON dict."""
    categories = [
        {"id": info["id"], "name": info["name"], "supercategory": "skin"}
        for info in CATEGORY_MAP.values()
    ]

    images = [{
        "id": image_id,
        "file_name": filename,
        "width": width,
        "height": height,
    }]

    annotations = []
    ann_id = 1
    for ann_info in all_annotations:
        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": ann_info["category_id"],
            "bbox": ann_info["bbox"],
            "area": ann_info["area"],
            "iscrowd": 0,
            "segmentation": ann_info["segmentation"],
        })
        ann_id += 1

    coco = {
        "info": {
            "description": "Skin single-patient segmentation dataset for RF-DETR",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "categories": categories,
        "annotations": annotations,
    }
    return coco


def visualize_single_category(image: np.ndarray, anns: list, color: tuple, out_path: Path):
    """Draw polygons for a single category onto the image and save."""
    vis = image.copy()
    for ann in anns:
        seg = ann["segmentation"][0]
        pts = np.array([[seg[i], seg[i + 1]] for i in range(0, len(seg), 2)], dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=2)
        cv2.fillPoly(vis, [pts], color=color)
    cv2.imwrite(str(out_path), vis)


def visualize_all_categories(image: np.ndarray, all_anns: list, out_path: Path):
    """Draw all category polygons with different colors."""
    vis = image.copy()
    # Predefined distinct colors (BGR)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128), (255, 128, 128), (128, 128, 255),
    ]

    for idx, ann in enumerate(all_anns):
        seg = ann["segmentation"][0]
        pts = np.array([[seg[i], seg[i + 1]] for i in range(0, len(seg), 2)], dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        color = colors[(idx) % len(colors)]
        cv2.fillPoly(vis, [pts], color=color)
        cv2.polylines(vis, [pts], isClosed=True, color=(255, 255, 255), thickness=1)

    cv2.imwrite(str(out_path), vis)


def main():
    if not MAIN_IMAGE.exists():
        raise FileNotFoundError(f"Main image not found: {MAIN_IMAGE}")
    if not MASK_DIR.exists():
        raise FileNotFoundError(f"Mask directory not found: {MASK_DIR}")

    # Prepare directories
    splits = ["train", "valid", "test"]
    for split in splits:
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Resize and copy main image
    target_filename = f"{PATIENT_ID}.jpg"
    for split in splits:
        dst = OUT_DIR / split / target_filename
        resize_main_image(MAIN_IMAGE, dst, TARGET_SIZE)

    # Process masks
    all_annotations = []  # list of dict with keys: category_id, segmentation, bbox, area
    total_lesions = 0

    for mask_name, cat_info in CATEGORY_MAP.items():
        mask_path = MASK_DIR / mask_name
        if not mask_path.exists():
            print(f"  [SKIP] Mask not found: {mask_path}")
            continue

        anns, lesion_count = extract_polygons_from_mask(mask_path, TARGET_SIZE, mask_name=mask_name)
        print(f"  {mask_name}: {len(anns)} polygons extracted, {lesion_count} lesion components")
        total_lesions += lesion_count

        for ann in anns:
            all_annotations.append({
                "category_id": cat_info["id"],
                **ann,
            })

    print(f"\nTotal lesion connected components across all masks: {total_lesions}")

    # Build COCO JSON
    coco = build_coco_json(
        image_id=1,
        filename=target_filename,
        width=TARGET_SIZE[0],
        height=TARGET_SIZE[1],
        all_annotations=all_annotations,
    )

    # Write JSON for all splits
    for split in splits:
        json_path = OUT_DIR / split / "_annotations.coco.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2, ensure_ascii=False)
        print(f"Written {json_path}")

    # Load resized image for visualization
    main_resized_path = OUT_DIR / "train" / target_filename
    vis_image = cv2.imread(str(main_resized_path))
    if vis_image is None:
        raise ValueError("Failed to load resized main image for visualization")
    vis_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)

    # Per-category visualizations
    for cat_id in range(1, 19):
        cat_anns = [a for a in all_annotations if a["category_id"] == cat_id]
        if not cat_anns:
            continue
        cat_name = next(v["name"] for v in CATEGORY_MAP.values() if v["id"] == cat_id)
        out_path = VIS_DIR / f"mask_{cat_name}.png"
        # Use a semi-transparent overlay approach: draw on copy, then blend
        overlay = vis_rgb.copy()
        color = (0, 255, 0)  # green in RGB
        for ann in cat_anns:
            seg = ann["segmentation"][0]
            pts = np.array([[seg[i], seg[i + 1]] for i in range(0, len(seg), 2)], dtype=np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color=color)
            cv2.polylines(overlay, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        blended = cv2.addWeighted(vis_rgb, 0.6, overlay, 0.4, 0)
        cv2.imwrite(str(out_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        print(f"Visualization saved: {out_path}")

    # Full overlay visualization
    all_overlay = vis_rgb.copy()
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128), (255, 128, 128), (128, 128, 255),
    ]
    for idx, ann in enumerate(all_annotations):
        seg = ann["segmentation"][0]
        pts = np.array([[seg[i], seg[i + 1]] for i in range(0, len(seg), 2)], dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        color = colors[(idx) % len(colors)]
        cv2.fillPoly(all_overlay, [pts], color=color)
    all_blended = cv2.addWeighted(vis_rgb, 0.5, all_overlay, 0.5, 0)
    all_out = VIS_DIR / "all_masks.png"
    cv2.imwrite(str(all_out), cv2.cvtColor(all_blended, cv2.COLOR_RGB2BGR))
    print(f"Full overlay saved: {all_out}")

    print("\nDone! Dataset ready at:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
