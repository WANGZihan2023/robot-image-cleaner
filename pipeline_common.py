"""机器人图像清洗 pipeline 的公共工具模块.

统一 CLI 参数, 路径解析, JSON 读写, 以及按 episode 分组排序的逻辑.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# 从 frame_000 这类文件名中提取帧序号, 用于按时间顺序排序
FRAME_PATTERN = re.compile(r"frame_(\d+)", re.IGNORECASE)
# 下游模型常用的输入尺寸
OUTPUT_SIZE = (224, 224)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """为各脚本注册统一的命令行参数, 保证 pipeline 各阶段路径/阈值一致."""
    parser.add_argument("--raw-dir", type=Path, default=Path("raw_data"))
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"))
    parser.add_argument("--intermediate-dir", type=Path, default=Path("intermediate"))
    parser.add_argument("--blur-threshold", type=float, default=100.0)
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="SSIM threshold; frames with score >= threshold are duplicates.",
    )


def parse_frame_index(frame_id: str) -> int:
    """从 frame_id 字符串中解析出整数帧序号."""
    match = FRAME_PATTERN.search(frame_id)
    if not match:
        raise ValueError(f"Invalid frame id: {frame_id}")
    return int(match.group(1))


def frame_sort_key(frame_id: str) -> int:
    """返回帧序号, 避免字符串排序导致 frame_10 排在 frame_2 前面."""
    return parse_frame_index(frame_id)


def load_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件并返回字典."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """将字典写入 JSON 文件, 自动创建父目录."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def iter_episode_images(raw_dir: Path) -> list[tuple[str, str, Path]]:
    """扫描 raw_data 下所有 jpg, 解析出 (episode_id, frame_id, 路径)."""
    records: list[tuple[str, str, Path]] = []
    for image_path in sorted(raw_dir.rglob("*.jpg")):
        episode_id = image_path.parent.name
        frame_id = image_path.stem
        records.append((episode_id, frame_id, image_path))
    return records


def group_by_episode(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """将扁平记录按 episode 分组, 并在组内按帧序号升序排列."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["episode_id"], []).append(record)
    for episode_records in grouped.values():
        episode_records.sort(key=lambda item: frame_sort_key(item["frame_id"]))
    return grouped
