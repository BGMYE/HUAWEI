# 钻孔成像裂隙自动识别方案

该项目实现了针对钻孔成像展开图的裂隙像素自动识别流程，覆盖 **预处理、特征提取、分类、模型评估和二值化分析** 全部环节。核心算法以 `fracture_detection` Python 包的形式实现，并提供可执行的命令行接口。

## 功能概述

1. **预处理**：对图像执行去噪和自适应对比度增强（高斯滤波、双边滤波、CLAHE），提升裂隙与围岩的对比度。
2. **特征提取**：计算梯度、拉普拉斯、局部熵、局部标准差以及多组 Gabor 滤波响应，构建高维像素特征向量。
3. **分类**：基于无监督的 `KMeans` 对像素聚类，通过梯度与亮度统计量自动判定裂隙簇，实现裂隙/背景分类。
4. **后处理与二值化**：使用形态学运算去除噪点、闭合间断，得到最终的裂隙二值掩膜并保存为 `PNG`。
5. **模型评估**：若提供人工标注的二值掩膜，可自动计算 `Accuracy / Precision / Recall / F1 / IoU` 等指标，输出 `metrics.csv`。

## 安装依赖

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 命令行使用方法

```bash
python -m fracture_detection.pipeline \
  --images data/raw_images \
  --masks data/ground_truth_masks \
  --output outputs \
  --save-intermediate
```

- `--images` 支持指定单张图片或包含图片的目录（`png/jpg/tif/bmp`）。
- `--masks` 可选，用于提供与图片同名（文件名 stem 相同）的二值标注图像。
- `--output` 指定预测掩膜与评估指标的输出目录（默认 `outputs`）。
- `--config` 可提供 JSON 文件覆盖默认参数（见下文配置项）。
- `--save-intermediate` 将保存预处理结果与原始/清理前掩膜的 `npy` 数组，便于调试。

命令运行后将输出：

- `*_mask.png`：裂隙二值掩膜（255 表示裂隙，0 表示背景）。
- `metrics.csv`：若存在标注掩膜，保存逐图像的评估指标。
- `intermediate/`：在启用 `--save-intermediate` 时保存的中间数据。

## 配置项

`PipelineConfig` 支持以下关键参数，可通过 `--config config.json` 进行覆盖：

| 参数 | 含义 | 默认值 |
| --- | --- | --- |
| `gaussian_sigma` | 预处理高斯滤波标准差 | 1.2 |
| `bilateral_sigma_color` / `bilateral_sigma_spatial` | 双边滤波颜色/空间参数 | 0.1 / 3.0 |
| `clahe_clip_limit` | CLAHE 对比度限制 | 0.03 |
| `entropy_disk_radius` | 局部熵统计半径（像素） | 5 |
| `gabor_frequencies` / `gabor_thetas` | Gabor 滤波频率与方向集合 | (0.1, 0.2, 0.3) / 0°~150° |
| `remove_small_objects` | 去除面积小于该像素数的连通域 | 200 |
| `closing_disk_radius` / `opening_disk_radius` | 闭运算与开运算结构元半径 | 2 / 1 |
| `kmeans_n_init` | KMeans 初始化次数 | 10 |

根据岩心材质、光照条件或标注粒度，可调整上述参数以获得更稳定的裂隙识别结果。

## 测试

```bash
pytest
```

单元测试在合成的裂隙图像上验证了管线的整体流程，确保裂隙区域被成功检测并具有良好的召回率。

## 目录结构

```
src/fracture_detection/    # 核心算法实现
├── __init__.py
└── pipeline.py            # 预处理、特征提取、分类、后处理、评估与 CLI

tests/                     # 单元测试
└── test_pipeline.py
```

## 参考与后续工作

- 若需要引入监督学习，可将 `classify_pixels` 替换为支持标注数据的分类器（如随机森林、轻量级 CNN），并利用 `evaluate_predictions` 统一评估。
- 可结合领域知识增加地质特征（如裂缝倾角、宽度统计）分析模块，进一步支持地质解释与报告生成。
