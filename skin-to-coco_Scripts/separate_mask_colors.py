#!/usr/bin/env python3
"""
Separate each mask into exactly 2 (or 3 for special masks) dominant colors.

Strategy:
1. Load mask as BGR.
2. Quantize BGR to coarse bins (8 per channel = 512 total).
3. Find unique coarse colors and counts.
4. Filter out near-black bins.
5. Weighted KMeans on coarse color centers (sqrt count as weight).
6. Map ALL foreground pixels to their cluster.
7. For each cluster, compute median BGR as representative value.
8. Build binary mask for each cluster.
9. For 3-color masks, split one cluster with another K=2.
10. Output masks named by median BGR tuple.

Usage:
    python skin-to-coco_Scripts/separate_mask_colors.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans

PATIENT_DIR = Path("data/33732019834458734")
MASK_DIR = PATIENT_DIR / "mask"
OUT_DIR = Path("visualizations/separated_masks")
REPORT_JSON = Path("skin-to-coco_Scripts/mask_dominant_colors.json")

THREE_COLOR_MASKS = {"Pore_Face.jpg", "Comedones_Face.jpg"}

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


def cluster_coarse_hsv(img: np.ndarray, n_clusters: int = 2, coarse_bins: int = 8):
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
    # H [0,179] -> bins, S [0,255] -> bins, V [0,255] -> bins
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


def separate_mask(mask_path: Path, out_dir: Path):
    """Separate a single mask into dominant color layers."""
    img = cv2.imread(str(mask_path))
    if img is None:
        raise ValueError(f"Cannot read mask: {mask_path}")

    h, w = img.shape[:2]

    # Main KMeans K=2 on coarse HSV
    labels, centers_bgr, ys, xs = cluster_coarse_hsv(img, n_clusters=2, coarse_bins=8)

    if len(labels) == 0:
        return []

    # Build clusters
    clusters = []
    for i in range(2):
        cluster_mask = (labels == i)
        cluster_bgr = img[ys[cluster_mask], xs[cluster_mask]]
        median_bgr = tuple(int(v) for v in np.median(cluster_bgr, axis=0)) if len(cluster_bgr) > 0 else centers_bgr[i]

        binary = np.zeros((h, w), dtype=np.uint8)
        binary[ys[cluster_mask], xs[cluster_mask]] = 255

        num_l, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        num_components = max(0, num_l - 1)
        total_pixels = int(np.sum(cluster_mask))

        clusters.append({
            "label": i,
            "median_bgr": median_bgr,
            "binary": binary,
            "num_components": num_components,
            "total_pixels": total_pixels,
        })

    # Sort by component count (fewer first)
    clusters.sort(key=lambda c: c["num_components"])

    rows = []

    # Save each cluster as a separate mask
    for cluster in clusters:
        bgr = cluster["median_bgr"]
        safe_name = f"{mask_path.stem}_color_{bgr[0]}_{bgr[1]}_{bgr[2]}.png"
        out_path = out_dir / safe_name
        cv2.imwrite(str(out_path), cluster["binary"])

        rows.append({
            "cluster": cluster["label"],
            "color_bgr": bgr,
            "pixel_count": cluster["total_pixels"],
            "num_components": cluster["num_components"],
            "output_file": str(out_path),
        })

    # For 3-color masks, split the second cluster (usually lesion with more components)
    if mask_path.name in THREE_COLOR_MASKS and len(clusters) >= 2:
        target_cluster = clusters[1]
        target_label = target_cluster["label"]
        target_mask = (labels == target_label)

        target_ys = ys[target_mask]
        target_xs = xs[target_mask]

        if len(target_ys) > 0:
            # Create lesion-only image for sub-clustering
            lesion_img = np.zeros_like(img)
            lesion_img[target_ys, target_xs] = img[target_ys, target_xs]
            sub_labels, sub_centers, sub_ys, sub_xs = cluster_coarse_hsv(lesion_img, n_clusters=2, coarse_bins=8)

            if len(sub_labels) > 0:
                for j in range(2):
                    sub_bgr = sub_centers[j]
                    binary = np.zeros((h, w), dtype=np.uint8)
                    binary[sub_ys[sub_labels == j], sub_xs[sub_labels == j]] = 255

                    num_l, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                    num_components = max(0, num_l - 1)

                    safe_name = f"{mask_path.stem}_color_{sub_bgr[0]}_{sub_bgr[1]}_{sub_bgr[2]}.png"
                    out_path = out_dir / safe_name
                    cv2.imwrite(str(out_path), binary)

                    rows.append({
                        "cluster": f"{target_label}.{j}",
                        "color_bgr": sub_bgr,
                        "pixel_count": int(np.sum(sub_labels == j)),
                        "num_components": num_components,
                        "output_file": str(out_path),
                    })

    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_report = {}

    for mask_name in CATEGORY_ORDER:
        mask_path = MASK_DIR / mask_name
        if not mask_path.exists():
            print(f"[MISSING] {mask_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Mask: {mask_name}")
        print(f"{'='*60}")

        rows = separate_mask(mask_path, OUT_DIR)
        if not rows:
            continue

        print(f"  {'cluster':>8} {'BGR':>15} {'pixels':>10} {'components':>10}")
        for r in rows:
            bgr = r["color_bgr"]
            print(
                f"  {str(r['cluster']):>8} ({bgr[0]:3d},{bgr[1]:3d},{bgr[2]:3d})"
                f"  {r['pixel_count']:>10} {r['num_components']:>10}"
            )

        all_report[mask_name] = rows

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_report, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Separated masks saved to: {OUT_DIR}")
    print(f"Report saved to: {REPORT_JSON}")


if __name__ == "__main__":
    main()
