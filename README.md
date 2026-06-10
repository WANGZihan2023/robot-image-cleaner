# Robot Image Cleaner

机器人图像数据清洗 pipeline：从模拟/真实采集的原始帧中自动剔除模糊图与相邻重复帧，统一 resize 到 224×224，并输出 metadata 与统计报告。

## 清洗规则

| 规则 | 方法 | 默认阈值 |
|------|------|----------|
| 模糊检测 | OpenCV Laplacian 方差 | `< 100.0` 判为 blur |
| 重复帧过滤 | 同 episode 内相邻保留帧 SSIM | `>= 0.95` 判为 duplicate |
| 输出尺寸 | `cv2.resize` | 224 × 224 |

## 目录结构

```text
├── raw_data/              # 原始图像（episode_xxx/frame_xxx.jpg）
├── processed/             # 清洗后输出
├── intermediate/          # 阶段间 JSON 结果
├── generate_mock_data.py  # 生成 ~1000 张模拟数据
├── filter_blur.py         # 模糊检测
├── dedup_frames.py        # 去重 + resize + 保存
├── build_metadata.py      # 汇总 metadata.csv
├── run_pipeline.py        # 一键跑全流程
├── metadata.csv           # 最终 metadata
└── reports/               # 可选统计图
```

## 环境配置（Conda）

在项目根目录创建本地 Conda 环境并安装依赖：

```bash
conda create -p .conda python=3.11 -y
conda run -p .conda pip install -r requirements.txt
```

或使用 `environment.yml`：

```bash
conda env create -p .conda -f environment.yml
```

激活环境：

```bash
conda activate ./.conda
```

## 快速开始

一键运行（生成模拟数据 + 清洗 + metadata + 统计图）：

```bash
conda activate ./.conda
python run_pipeline.py --plot
```

已有 `raw_data/` 时跳过生成：

```bash
python run_pipeline.py --skip-generate --plot
```

分步运行：

```bash
python generate_mock_data.py
python filter_blur.py
python dedup_frames.py
python build_metadata.py --plot
```

## 统计报告（实测）

```text
原始帧数:       1000
保留帧数:        820
剔除 - 模糊:     126
剔除 - 重复:      54
废片率降低:      18.0%
模糊检测速度:    476.87 张/秒
去重阶段速度:    170.95 张/秒
```

## metadata.csv 字段

| 列 | 说明 |
|----|------|
| `episode_id` | 如 `episode_001` |
| `frame_id` | 如 `frame_000` |
| `file_path` | 原始路径 |
| `blur_score` | Laplacian 方差 |
| `is_valid` | 是否保留 |
| `reason` | 空 / `blur` / `duplicate` |

## 简历描述参考

> 搭建机器人图像数据清洗 pipeline，基于 OpenCV Laplacian 方差实现模糊检测，结合 SSIM 完成 episode 内相邻重复帧过滤；模块化设计 4 个脚本 + 一键编排，处理效率 **477 张/秒**，废片率降低 **18%**（1000→820）。

## 技术栈

- Python：`pathlib`, `json`, `argparse`
- OpenCV：读图、resize、Laplacian 模糊检测
- NumPy：数组运算
- scikit-image：SSIM
- Pandas：metadata.csv
- Matplotlib：清洗前后对比图（可选）
