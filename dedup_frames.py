#!/usr/bin/env python3
"""去重阶段: 在 blur 通过的帧中, 按 episode 内时间顺序剔除相邻重复帧.

重复判定: 当前帧与上一个保留帧的 SSIM >= 阈值.
通过后 resize 到 224x224 并写入 processed/.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import cv2
from skimage.metrics import structural_similarity as ssim

from pipeline_common import OUTPUT_SIZE, add_common_args, load_json, save_json


def read_and_resize(image_path: Path) -> tuple:
    """读取原图并缩放到统一尺寸, 同时返回灰度图用于 SSIM 比较."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    resized = cv2.resize(image, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return resized, gray


def compute_ssim(gray_a, gray_b, similarity_threshold: float) -> float:
    """计算两帧灰度图的 SSIM 相似度, 返回值范围 [0, 1]."""
    score = float(ssim(gray_a, gray_b, data_range=255))
    return score


def dedup_frames(
    raw_dir: Path,
    processed_dir: Path,
    intermediate_dir: Path,
    similarity_threshold: float,
) -> dict:
    """对 blur 通过的帧做相邻去重, resize 后保存到 processed/, 并写入 dedup_results.json."""
    blur_path = intermediate_dir / "blur_results.json"
    if not blur_path.exists():
        raise FileNotFoundError(f"Missing blur results: {blur_path}")

    blur_results = load_json(blur_path)
    # 重建输出目录, 避免多次运行残留旧文件
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    dedup_results: dict[str, list[dict]] = {}
    total_candidates = 0
    duplicate_count = 0
    kept_count = 0

    for episode_id in sorted(blur_results.keys()):
        episode_records = blur_results[episode_id]
        dedup_results[episode_id] = []
        last_kept_gray = None
        last_kept_resized = None

        for record in episode_records:
            result = dict(record)
            if not record["is_valid"]:
                # 已在 blur 阶段剔除, 直接透传
                dedup_results[episode_id].append(result)
                continue

            total_candidates += 1
            image_path = Path(record["file_path"])
            resized, gray = read_and_resize(image_path)

            is_duplicate = False
            if last_kept_gray is not None:
                # 与上一个保留帧比较, 而非上一原始帧 (中间可能有 blur 剔除)
                score = compute_ssim(last_kept_gray, gray, similarity_threshold)
                if score >= similarity_threshold:
                    is_duplicate = True
                    result["is_valid"] = False
                    result["reason"] = "duplicate"
                    result["ssim_score"] = round(score, 4)
                    duplicate_count += 1

            if is_duplicate:
                dedup_results[episode_id].append(result)
                continue

            output_dir = processed_dir / episode_id
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{record['frame_id']}.jpg"
            cv2.imwrite(str(output_path), resized)

            result["processed_path"] = str(output_path)
            dedup_results[episode_id].append(result)
            last_kept_gray = gray
            last_kept_resized = resized
            kept_count += 1

    elapsed = time.perf_counter() - start
    save_json(intermediate_dir / "dedup_results.json", dedup_results)

    stats = {
        "blur_passed_candidates": total_candidates,
        "rejected_duplicate": duplicate_count,
        "kept": kept_count,
        "elapsed_seconds": round(elapsed, 4),
        "images_per_second": round(total_candidates / elapsed if elapsed > 0 else 0.0, 2),
    }
    save_json(intermediate_dir / "dedup_stats.json", stats)

    print("Dedup complete")
    print(f"Blur-passed candidates: {total_candidates}")
    print(f"Rejected (duplicate): {duplicate_count}")
    print(f"Kept and saved: {kept_count}")
    print(f"Speed: {stats['images_per_second']:.2f} images/sec")
    return stats


def main() -> None:
    """命令行入口: 解析参数并执行去重与 resize 保存."""
    parser = argparse.ArgumentParser(description="Deduplicate adjacent frames within episodes.")
    add_common_args(parser)
    args = parser.parse_args()
    dedup_frames(
        args.raw_dir,
        args.processed_dir,
        args.intermediate_dir,
        args.similarity_threshold,
    )


if __name__ == "__main__":
    main()
