#!/usr/bin/env python3
"""生成模拟机器人采集图像, 并注入模糊帧与重复帧用于验证清洗 pipeline.

输出结构: raw_data/episode_XXX/frame_YYY.jpg
调试辅助: intermediate/mock_manifest.json 记录故意注入的废片位置 (不参与最终 metadata).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from pipeline_common import add_common_args, save_json

IMAGE_SIZE = (640, 480)
NUM_EPISODES = 10
FRAMES_PER_EPISODE = 100
NUM_BLUR = 120       # 约 12% 模糊废片
NUM_DUPLICATE = 60   # 约 6% 重复废片, 合计剔除 ~18%


def build_base_frame(
    episode_idx: int, frame_idx: int, rng: random.Random, np_rng: np.random.Generator
) -> np.ndarray:
    """绘制一帧模拟相机画面: 随机几何图形 + 噪声 + episode/frame 水印."""
    image = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 30, dtype=np.uint8)
    for _ in range(rng.randint(4, 8)):
        color = tuple(int(v) for v in rng.sample(range(80, 255), 3))
        if rng.random() < 0.5:
            center = (rng.randint(40, IMAGE_SIZE[0] - 40), rng.randint(40, IMAGE_SIZE[1] - 40))
            radius = rng.randint(20, 90)
            cv2.circle(image, center, radius, color, -1)
        else:
            x1 = rng.randint(0, IMAGE_SIZE[0] - 80)
            y1 = rng.randint(0, IMAGE_SIZE[1] - 80)
            x2 = x1 + rng.randint(40, 160)
            y2 = y1 + rng.randint(40, 120)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

    # 加入传感器噪声, 使每帧略有差异
    noise = np_rng.integers(0, 26, size=image.shape, dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    label = f"ep{episode_idx:03d}-f{frame_idx:03d}"
    cv2.putText(
        image,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def apply_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """高斯模糊, 模拟运动/对焦失败导致的废片."""
    kernel = rng.choice([15, 17, 19, 21, 25, 31])
    if kernel % 2 == 0:
        kernel += 1  # OpenCV 要求核尺寸为奇数
    return cv2.GaussianBlur(image, (kernel, kernel), 0)


def choose_injection_indices(total_frames: int, count: int, rng: random.Random) -> list[int]:
    """从全部帧中随机选取 count 个位置, 用于注入废片."""
    return sorted(rng.sample(range(total_frames), count))


def generate_mock_data(raw_dir: Path, intermediate_dir: Path, seed: int = 42) -> dict:
    """生成 1000 张模拟图像, 注入模糊/重复废片, 并写入 mock_manifest.json."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # 每次重新生成时清空旧数据, 保证可复现
    if raw_dir.exists():
        for path in raw_dir.rglob("*"):
            if path.is_file():
                path.unlink()
    raw_dir.mkdir(parents=True, exist_ok=True)

    total_frames = NUM_EPISODES * FRAMES_PER_EPISODE
    blur_indices = set(choose_injection_indices(total_frames, NUM_BLUR, rng))
    # 重复帧必须复制前一帧, 因此候选 index 从 1 开始, 且不与 blur 重叠
    duplicate_candidates = [idx for idx in range(1, total_frames) if idx not in blur_indices]
    duplicate_indices = set(rng.sample(duplicate_candidates, NUM_DUPLICATE))

    manifest = {
        "seed": seed,
        "total_frames": total_frames,
        "blur_frames": [],
        "duplicate_frames": [],
    }

    flat_index = 0
    previous_image: np.ndarray | None = None

    for episode_idx in range(1, NUM_EPISODES + 1):
        episode_id = f"episode_{episode_idx:03d}"
        episode_dir = raw_dir / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in range(FRAMES_PER_EPISODE):
            frame_id = f"frame_{frame_idx:03d}"
            image_path = episode_dir / f"{frame_id}.jpg"

            if flat_index in duplicate_indices and previous_image is not None:
                # 直接复制上一帧, 保证 SSIM > 0.95
                image = previous_image.copy()
                manifest["duplicate_frames"].append(
                    {
                        "episode_id": episode_id,
                        "frame_id": frame_id,
                        "flat_index": flat_index,
                    }
                )
            else:
                image = build_base_frame(episode_idx, frame_idx, rng, np_rng)
                image = image + np_rng.integers(-2, 3, size=image.shape, dtype=np.int16)
                image = np.clip(image, 0, 255).astype(np.uint8)

            if flat_index in blur_indices:
                image = apply_blur(image, rng)
                manifest["blur_frames"].append(
                    {
                        "episode_id": episode_id,
                        "frame_id": frame_id,
                        "flat_index": flat_index,
                    }
                )

            cv2.imwrite(str(image_path), image)
            previous_image = image
            flat_index += 1

    save_json(intermediate_dir / "mock_manifest.json", manifest)
    return manifest


def main() -> None:
    """命令行入口: 解析参数并执行模拟数据生成."""
    parser = argparse.ArgumentParser(description="Generate synthetic robot image data.")
    add_common_args(parser)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = generate_mock_data(args.raw_dir, args.intermediate_dir, seed=args.seed)
    print(f"Generated {manifest['total_frames']} frames in {args.raw_dir}")
    print(f"Injected blur frames: {len(manifest['blur_frames'])}")
    print(f"Injected duplicate frames: {len(manifest['duplicate_frames'])}")


if __name__ == "__main__":
    main()
