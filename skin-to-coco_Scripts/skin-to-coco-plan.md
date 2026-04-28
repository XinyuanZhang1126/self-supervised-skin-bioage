# 皮肤数据集 → RF-DETR-Seg COCO 格式转换计划

## 目标
将单患者皮肤数据 `data/33732019834458734/` 转换为 RF-DETR-Seg 适用的 COCO 分割数据集，输出到 `data/skin_single_patient/`。

## 输入输出目录结构

### 输入目录（原始数据）
```
data/33732019834458734/
├── Rgb.jpg                 # 主图（原始分辨率）
├── mask/
│   ├── Below_Eye_Wrinkles.jpg
│   ├── Between_Wrinkles.jpg
│   ├── Fish_tail_Wrinkles.jpg
│   ├── Forehead_Wrinkles.jpg
│   ├── Nasal_Wrinkles.jpg
│   ├── Nose_root_Wrinkles.jpg
│   ├── Acne_Face.jpg
│   ├── Bloodshot_Face.jpg
│   ├── Brown_Tag_Face.jpg
│   ├── Comedones_Face.jpg
│   ├── Dark_Face.jpg
│   ├── Pore_Face.jpg
│   ├── Rgb_Tag.jpg
│   ├── Senstive_Face.jpg
│   ├── T_Oil_Face.jpg
│   ├── U_Oil_Face.jpg
│   ├── Uv_Ex_Tag_Face.jpg
│   └── Uv_P_Face.jpg
└── report.json
```

### 输出目录（COCO 格式）
```
data/skin_single_patient/
├── train/
│   ├── _annotations.coco.json
│   └── 33732019834458734.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── 33732019834458734.jpg
└── test/
    ├── _annotations.coco.json
    └── 33732019834458734.jpg
```

---

## Category 映射表（18 类）

| category_id | mask 文件名 | category name |
|-------------|------------|---------------|
| 1 | Below_Eye_Wrinkles.jpg | Below_Eye_Wrinkles |
| 2 | Between_Wrinkles.jpg | Between_Wrinkles |
| 3 | Fish_tail_Wrinkles.jpg | Fish_tail_Wrinkles |
| 4 | Forehead_Wrinkles.jpg | Forehead_Wrinkles |
| 5 | Nasal_Wrinkles.jpg | Nasal_Wrinkles |
| 6 | Nose_root_Wrinkles.jpg | Nose_root_Wrinkles |
| 7 | Acne_Face.jpg | Acne_Face |
| 8 | Bloodshot_Face.jpg | Bloodshot_Face |
| 9 | Brown_Tag_Face.jpg | Brown_Tag_Face |
| 10 | Comedones_Face.jpg | Comedones_Face |
| 11 | Dark_Face.jpg | Dark_Face |
| 12 | Pore_Face.jpg | Pore_Face |
| 13 | Rgb_Tag.jpg | Rgb_Tag |
| 14 | Senstive_Face.jpg | Senstive_Face |
| 15 | T_Oil_Face.jpg | T_Oil_Face |
| 16 | U_Oil_Face.jpg | U_Oil_Face |
| 17 | Uv_Ex_Tag_Face.jpg | Uv_Ex_Tag_Face |
| 18 | Uv_P_Face.jpg | Uv_P_Face |

---

## 主图预处理逻辑

1. **读取原始主图**：`data/33732019834458734/Rgb.jpg`
2. **模式转换**：若图片模式为 `RGBA` 或 `P`，先转换为 `RGB`
3. **Resize**：使用 PIL 的 `Image.Resampling.LANCZOS` 插值算法，resize 到 `768×768`
4. **保存**：以 JPEG 格式保存，quality=95，文件名为 `33732019834458734.jpg`
5. **复制到三个 split 目录**：分别写入 `train/`、`valid/`、`test/`

---

## Mask → Segmentation 核心转换逻辑

### 整体流程
对每张 mask 图片执行以下步骤：

### 1. 加载 Mask
- 使用 `cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)` 读取为灰度图
- 获取原始尺寸 `orig_h × orig_w`
- 计算缩放比例：
  - `scale_x = target_w / orig_w`
  - `scale_y = target_h / orig_h`
  - 其中 `target_wh = (768, 768)`

### 2. 自适应颜色过滤（前景 vs ROI 大圈）
- 提取所有非零像素值的唯一值：`unique_vals = np.unique(img[img > 0])`
- **ROI 判定**：取最亮的非零值作为 ROI 大圈色
  - `roi_val = int(unique_vals.max())`
- **前景判定**：其余非零值均为前景
  - `foreground_vals = unique_vals[unique_vals != roi_val]`
- **边界情况**：若只有一个非零值，则将其全部视为前景（罕见情况）

### 3. 二值化与形态学去噪
- 构建前景二值 mask：
  - `binary_mask = np.isin(img, foreground_vals).astype(np.uint8) * 255`
- 形态学开运算去除微小噪声：
  - `kernel = np.ones((3, 3), np.uint8)`
  - `cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)`

### 4. 连通域分析与轮廓提取
- 查找外轮廓：`cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`
- 对每个轮廓：
  - 计算面积：`area = cv2.contourArea(cnt)`
  - **面积过滤**：若 `area < MIN_CONTOUR_AREA`（即 `4.0`），则丢弃
  - **多边形逼近**：
    - `epsilon = 0.005 * cv2.arcLength(cnt, True)`
    - `approx = cv2.approxPolyDP(cnt, epsilon, True)`
  - 若逼近后顶点数 `< 3`，则丢弃

### 5. 坐标缩放（关键）
所有 polygon、bbox、area 均按 `scale_x` / `scale_y` 缩放：

- **Polygon**：
  ```
  x = float(pt[0][0] * scale_x)
  y = float(pt[0][1] * scale_y)
  保留两位小数
  ```
- **BBox**：
  ```
  x, y, w, h = cv2.boundingRect(approx)
  bbox = [round(x * scale_x, 2), round(y * scale_y, 2),
          round(w * scale_x, 2), round(h * scale_y, 2)]
  ```
- **Area**：
  ```
  scaled_area = float(area * scale_x * scale_y)  # 保留两位小数
  ```

---

## COCO JSON 生成逻辑

由 `build_coco_json()` 函数生成，结构如下：

```json
{
  "info": {
    "description": "Skin single-patient segmentation dataset for RF-DETR",
    "version": "1.0"
  },
  "licenses": [],
  "images": [
    {
      "id": 1,
      "file_name": "33732019834458734.jpg",
      "width": 768,
      "height": 768
    }
  ],
  "categories": [
    {"id": 1, "name": "Below_Eye_Wrinkles", "supercategory": "skin"},
    ...
    {"id": 18, "name": "Uv_P_Face", "supercategory": "skin"}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": <category_id>,
      "bbox": [x, y, w, h],
      "area": <scaled_area>,
      "iscrowd": 0,
      "segmentation": [[x1, y1, x2, y2, ...]]
    },
    ...
  ]
}
```

- `annotation.id` 从 1 开始递增
- 每张 mask 可能产生多个 annotation（对应多个不连通的前景区域）
- 所有 polygon 均包装为 `segmentation: [[x1,y1,x2,y2,...]]`（单 polygon 列表）

---

## 数据集划分策略

由于当前数据仅包含 **单一患者**（`33732019834458734`），无法进行常规的数据划分：

- **train / valid / test 的内容完全相同**
- 三张 COCO JSON 文件内容完全一致
- 三张主图 `33732019834458734.jpg` 也完全相同
- 这是一个临时策略，后续增加多患者数据后可改为真正的随机划分

---

## 可视化验证输出

转换完成后，在 `visualizations/` 目录生成以下验证图片：

### 1. 逐类别可视化
- 文件路径：`visualizations/mask_{category_name}.png`
- 内容：将该 category 的所有 polygon 以 **绿色填充** + **红色轮廓** 叠加到 resize 后的主图上
- 使用 `cv2.addWeighted` 进行半透明混合（原图 0.6，叠加层 0.4）

### 2. 全类别叠加可视化
- 文件路径：`visualizations/all_masks.png`
- 内容：将 18 个 category 的所有 polygon 用 **18 种不同颜色** 叠加到同一张图上
- 每种颜色预定义（BGR 格式），同样使用 `cv2.addWeighted` 混合（原图 0.5，叠加层 0.5）

---

## 相关脚本文件清单及说明

| 脚本文件 | 说明 |
|---------|------|
| `convert_skin_to_coco.py` | **主转换脚本**。包含：主图 resize、mask 转 polygon、COCO JSON 生成、数据集复制、可视化输出。核心函数：`resize_main_image()`、`extract_polygons_from_mask()`、`build_coco_json()`、`visualize_single_category()`、`visualize_all_categories()` |
| `analyze_mask_colors.py` | **颜色分析脚本**。对每个 mask 统计所有非零像素值的分布（像素总数、连通域数量、最大连通域面积、集中度），并生成 `mask_color_analysis.txt` 报告及 `visualizations/analysis/` 下的逐值可视化图。用于验证自适应颜色过滤策略的正确性 |

---

## 关键常量汇总

| 常量 | 值 | 说明 |
|------|-----|------|
| `TARGET_SIZE` | `(768, 768)` | 主图 resize 目标尺寸 |
| `MIN_CONTOUR_AREA` | `4.0` | 最小轮廓面积阈值（resized 像素单位） |
| `epsilon` | `0.005 * arcLength` | `approxPolyDP` 逼近精度 |
| `kernel` | `np.ones((3,3), np.uint8)` | 形态学开运算核 |
| `morphologyEx` | `MORPH_OPEN`, `iterations=1` | 去噪操作 |
