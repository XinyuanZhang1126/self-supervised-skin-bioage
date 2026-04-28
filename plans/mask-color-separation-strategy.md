# 皮肤 Mask 颜色分离策略文档

## 1. 问题背景

皮肤数据的 mask 并非传统的灰度二值图，而是 **BGR 彩色图像**。由于以下原因，同一张 mask 中包含数千种不同颜色：

- **反锯齿（Anti-aliasing）**：FOV 边界环和 lesion 边缘存在渐变过渡色
- **JPEG 压缩**：色块之间产生压缩伪影
- **多语义图层**：同一张 mask 中同时包含 FOV（人脸边界环/背景环）和 lesions（病灶区域）

**核心需求**：将每张 mask 分离为 2~3 个主导颜色图层（二值掩膜），后续通过颜色值区分 FOV 与 lesion。

---

## 2. 失败尝试与原因分析

### Iteration 1：原始像素 KMeans（K=2）
- **做法**：对所有非黑 BGR 像素直接运行 KMeans(K=2)
- **失败原因**：反锯齿产生的数千种渐变颜色将聚类中心拉向中间灰色，导致同一簇内混合了 FOV 和 lesion 像素
- **典型错误**：`Acne_Face` 的两个分离结果中都同时包含紫色 FOV 和青色 lesion

### Iteration 2：最大连通域 = FOV
- **做法**：将最大连通域判定为 FOV（人脸大圈）
- **失败原因**：
  - FOV 不是实心圆，而是**细线构成的边界环/轮廓**
  - 某些 mask（如 `Below_Eye_Wrinkles`）有 **2~3 个不连通的 FOV 环**
  - 无法通过连通域大小可靠区分 FOV 与 lesion

### Iteration 3：HSV + S/V 硬过滤
- **做法**：读入 RGBA → alpha>0 → HSV → 过滤 S<30 或 V<30 的像素 → KMeans K=2
- **失败原因**：
  - 过滤条件过于激进，**删除了反锯齿像素**
  - 某些 lesion 本身就是低饱和度细线（如皱纹的像素级线条），被误删
- **用户反馈**："I don't want anti-aliasing structures completely deleted."

### Iteration 4：H-only 加权 KMeans
- **做法**：仅使用 H 通道，以 S×V 作为样本权重进行 KMeans
- **失败原因**：某些 mask 中 FOV 和 lesion 的 Hue 过于接近
  - `Brown_Tag_Face`：FOV 为紫色（H≈126），lesion 为红色（H≈120），H 差仅 6°，聚类失败
  - 红色 lesion 掩膜中残留紫色 FOV 细线

### Iteration 5：Coarse BGR 量化
- **做法**：BGR 各通道量化为 8 bins，在粗化后的 BGR 中心上做 KMeans
- **失败原因**：`Acne_Face` 的 lesion（青色）被合并入 FOV（紫色），因为 BGR 空间中青色与紫色不够分离

---

## 3. 当前最终策略：Coarse HSV Quantization + Weighted KMeans

### 3.1 核心思想

利用 **HSV 颜色空间** 的特性：
- **Hue（H）** 将不同颜色族（紫、红、青、绿）干净地分开
- **Saturation（S）/ Value（V）** 处理饱和度与亮度变化，但参与聚类而非用于硬过滤

通过 **粗化量化** 将反锯齿产生的数千种颜色压缩为几十~几百个粗化颜色中心，再在这些中心上做 KMeans，既保留了所有像素，又实现了语义层面的颜色分离。

### 3.2 算法步骤

```python
def cluster_coarse_hsv(img, n_clusters=2, coarse_bins=8):
    # Step 1: BGR -> HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Step 2: 提取所有非零像素
    ys, xs = np.where(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 0)
    pixels = hsv[ys, xs]

    # Step 3: Coarse quantize H, S, V
    h_quant = pixels[:, 0] // (180 // coarse_bins)   # H [0,179]
    s_quant = pixels[:, 1] // (256 // coarse_bins)   # S [0,255]
    v_quant = pixels[:, 2] // (256 // coarse_bins)   # V [0,255]

    # Step 4: 统计粗化颜色的唯一值与频数
    uniques, counts = np.unique(
        np.stack([h_quant, s_quant, v_quant], axis=1),
        axis=0, return_counts=True
    )

    # Step 5: 过滤纯黑/近黑（仅排除无效背景，不删除 lesion）
    non_black = (uniques[:, 2] >= 2) & (uniques[:, 1] >= 1)
    uniques = uniques[non_black]
    counts = counts[non_black]

    # Step 6: 计算粗化颜色中心（HSV）
    h_c = uniques[:, 0] * (180 // bins) + (180 // bins) // 2
    s_c = uniques[:, 1] * (256 // bins) + (256 // bins) // 2
    v_c = uniques[:, 2] * (256 // bins) + (256 // bins) // 2
    coarse_colors = np.stack([h_c, s_c, v_c], axis=1)

    # Step 7: 加权 KMeans（sqrt(count) 为权重）
    repeated = []
    for color, weight in zip(coarse_colors, np.sqrt(counts)):
        repeated.extend([color] * max(1, int(weight)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(np.array(repeated))

    # Step 8: 将粗化颜色映射回原始像素
    quant_to_cluster = {
        tuple(q): lbl for q, lbl in zip(uniques, kmeans.predict(coarse_colors))
    }
    labels = [quant_to_cluster.get(tuple(q), -1) for q in pixel_quants]

    # Step 9: HSV 中心转 BGR 用于命名/展示
    centers_bgr = [cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_HSV2BGR)[0,0]
                   for c in kmeans.cluster_centers_]
    return labels, centers_bgr, ys, xs
```

### 3.3 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `coarse_bins` | 8 | H/S/V 各量化为 8 档 |
| `n_clusters` | 2 | 普通 mask 聚为 2 色 |
| 权重函数 | `sqrt(count)` | 防止某一颜色因像素过多主导聚类 |
| 近黑过滤 | `V >= 2 bins` 且 `S >= 1 bin` | 仅排除纯黑背景，保留所有 lesion |

### 3.4 3-Color Mask 子聚类

`Pore_Face.jpg` 和 `Comedones_Face.jpg` 需要 3 个主导色：
1. 先整体 K=2 聚类
2. 取 component 数更多的簇（通常为 lesion 簇）再次运行 K=2 子聚类
3. 最终输出 4 张二值掩膜（含 FOV + 3 个 lesion 色，或经合并后为 3 个有效色）

---

## 4. 当前分离结果汇总

基于 coarse HSV 策略运行后，各 mask 的主导色及连通域统计：

| Mask | 颜色1 (BGR) | CC | 颜色2 (BGR) | CC | 颜色3/4 (BGR) | CC | 备注 |
|------|-------------|-----|-------------|-----|---------------|-----|------|
| Below_Eye_Wrinkles | (166,105,117) | 2 | (0,255,0) | 94 | — | — | 紫 / 绿 |
| Between_Wrinkles | (166,105,117) | 1 | (0,255,255) | 6 | — | — | 紫 / 青 |
| Fish_tail_Wrinkles | (166,105,117) | 2 | (0,255,0) | 45 | — | — | 紫 / 绿 |
| Forehead_Wrinkles | (166,105,117) | 1 | (0,255,255) | 77 | — | — | 紫 / 青 |
| Nasal_Wrinkles | (166,105,117) | 2 | (97,61,69) | 2152 | — | — | 紫 / 深褐 |
| Nose_root_Wrinkles | (166,105,117) | 1 | (0,255,0) | 73 | — | — | 紫 / 绿 |
| Acne_Face | (166,105,117) | 4 | (255,255,0) | 4 | — | — | 紫 / 青 |
| Bloodshot_Face | (166,105,117) | 1 | (0,0,255) | 128 | — | — | 紫 / 蓝 |
| Brown_Tag_Face | (166,105,117) | 1 | (255,0,0) | 318 | — | — | 紫 / 红 |
| Comedones_Face | (166,105,117) | 490 | (255,0,96) | 400 | (113,66,68) / (191,111,111) | 843 / 401 | 4色，子聚类结果 |
| Dark_Face | (166,105,117) | 5 | (95,185,154) | 19 | — | — | 紫 / 青绿 |
| Pore_Face | (166,105,117) | 64 | (255,255,0) | 3208 | (222,92,17) / (115,43,8) | 3317 / 34135 | 4色，子聚类结果 |
| Rgb_Tag | (166,105,117) | 1 | (0,215,255) | 43 | — | — | 紫 / 浅青 |
| Senstive_Face | (166,105,117) | 1 | (255,255,0) | 308 | — | — | 紫 / 青 |
| T_Oil_Face | (166,105,117) | 1 | (0,129,0) | 218 | — | — | 紫 / 深绿 |
| U_Oil_Face | (166,105,117) | 1 | (0,84,0) | 215 | — | — | 紫 / 绿 |
| Uv_Ex_Tag_Face | (166,105,117) | 1 | (0,165,255) | 625 | — | — | 紫 / 橙 |
| Uv_P_Face | (166,105,117) | 2 | (0,255,0) | 1830 | — | — | 紫 / 绿 |

> **CC** = Connected Components（连通域数量）。FOV 通常为 1~2 个连通域，lesion 通常数量较多。

### 关键观察
- **紫色 (166,105,117)** 在几乎所有 mask 中都作为一个主导色出现，且连通域数量极少（1~5），高度疑似 FOV。
- 其余颜色（青、红、绿、蓝、橙等）连通域数量多，高度疑似 lesion 区域。

---

## 5. 相关脚本文件

| 脚本 | 路径 | 说明 |
|------|------|------|
| `separate_mask_colors.py` | `skin-to-coco_Scripts/separate_mask_colors.py` | 颜色分离主脚本，输出 `visualizations/separated_masks/` 及 `mask_dominant_colors.json` |
| `convert_skin_to_coco.py` | `skin-to-coco_Scripts/convert_skin_to_coco.py` | COCO 转换脚本，内含相同的 `_cluster_coarse_hsv()` 函数 |

---

## 6. TODO List（待完成）

- [ ] **用户确认 Comedones_Face 4 色分离结果**
  - 检查 `Comedones_Face_color_166_105_117.png`（疑似 FOV）
  - 检查 `Comedones_Face_color_255_0_96.png`、`Comedones_Face_color_113_66_68.png`、`Comedones_Face_color_191_111_111.png`（疑似 lesions）
  - 确认 4 个颜色是否符合视觉预期，是否需要合并某些 lesion 色

- [ ] **用户确认 Pore_Face 4 色分离结果**
  - 检查 `Pore_Face_color_166_105_117.png`（疑似 FOV）
  - 检查 `Pore_Face_color_255_255_0.png`、`Pore_Face_color_222_92_17.png`、`Pore_Face_color_115_43_8.png`（疑似 lesions）
  - 注意 `Pore_Face_color_115_43_8.png` 连通域高达 34135，需确认是否属于噪声或微小点

- [ ] **用户手动指定各 mask 类型的 FOV / Lesion 颜色映射**
  - 确认每种 mask 中哪个 BGR 颜色代表 FOV（紫色 166,105,117 已被高度怀疑）
  - 确认每种 mask 中哪个/哪些 BGR 颜色代表 lesion
  - 将映射录入 `convert_skin_to_coco.py` 的 `MASK_COLOR_CONFIG` 字典

- [ ] **运行 COCO 转换并验证最终标注**
  - 在 `MASK_COLOR_CONFIG` 填充后，重新运行 `convert_skin_to_coco.py`
  - 检查 `visualizations/mask_*.png` 确保 lesion 区域正确、FOV 已被排除
  - 统计各 category 的 lesion 数量（连通组件数）是否符合预期

- [ ] **多患者泛化验证**
  - 当前颜色映射基于单患者（`33732019834458734`）建立
  - 未来新增患者时，验证相同 mask 类型的颜色是否一致
  - 若颜色出现偏移，考虑在 `MASK_COLOR_CONFIG` 中引入 HSV 范围而非精确 BGR 值
