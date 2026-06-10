#!/usr/bin/env python3
"""模糊检测阶段: 用 Laplacian 方差评估图像清晰度.

方差越低说明边缘越弱, 图像越模糊. 本阶段只打标, 不删除原图.
结果写入 intermediate/blur_results.json, 供去重与 metadata 汇总使用.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from pipeline_common import add_common_args, group_by_episode, iter_episode_images, save_json


def compute_blur_score(image_path: Path) -> float:
    """计算 Laplacian 方差作为 blur_score, 值越大越清晰."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def filter_blur(raw_dir: Path, intermediate_dir: Path, blur_threshold: float) -> dict:
    """扫描全部原始帧, 计算 blur_score 并写入 blur_results.json 与 blur_stats.json."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    start = time.perf_counter()
    flat_records: list[dict] = []

    for episode_id, frame_id, image_path in iter_episode_images(raw_dir):
        blur_score = compute_blur_score(image_path)
        is_valid = blur_score >= blur_threshold
        flat_records.append(
            {
                "episode_id": episode_id,
                "frame_id": frame_id,
                "file_path": str(image_path),
                "blur_score": round(blur_score, 4),
                "is_valid": is_valid,
                "reason": "" if is_valid else "blur",
            }
        )

    elapsed = time.perf_counter() - start
    grouped = group_by_episode(flat_records)
    save_json(intermediate_dir / "blur_results.json", grouped)

    total = len(flat_records)
    rejected = sum(1 for record in flat_records if not record["is_valid"])
    speed = total / elapsed if elapsed > 0 else 0.0

    stats = {
        "total": total,
        "rejected_blur": rejected,
        "passed": total - rejected,
        "elapsed_seconds": round(elapsed, 4),
        "images_per_second": round(speed, 2),
    }
    save_json(intermediate_dir / "blur_stats.json", stats)

    print("Blur filter complete")
    print(f"Total frames: {total}")
    print(f"Rejected (blur): {rejected}")
    print(f"Passed: {total - rejected}")
    print(f"Speed: {speed:.2f} images/sec")
    return stats


def main() -> None:
    """命令行入口: 解析参数并执行模糊检测."""
    parser = argparse.ArgumentParser(description="Filter blurry robot camera frames.")
    add_common_args(parser)
    args = parser.parse_args()
    filter_blur(args.raw_dir, args.intermediate_dir, args.blur_threshold)


if __name__ == "__main__":
    main()
