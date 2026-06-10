#!/usr/bin/env python3
"""一键编排完整清洗 pipeline.

执行顺序: generate_mock_data -> filter_blur -> dedup_frames -> build_metadata
各子脚本也可独立运行, 便于调试和模块化展示.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from pipeline_common import add_common_args, load_json


def run_step(command: list[str]) -> None:
    """以子进程方式执行单个 pipeline 步骤, 失败时抛出异常."""
    print(f"\n>>> {' '.join(command)}")
    subprocess.run(command, check=True)


def run_pipeline(
    raw_dir: Path,
    processed_dir: Path,
    intermediate_dir: Path,
    blur_threshold: float,
    similarity_threshold: float,
    skip_generate: bool,
    seed: int,
    plot: bool,
    output_csv: Path,
) -> None:
    """按顺序串联生成数据, 模糊检测, 去重, metadata 四个阶段."""
    script_dir = Path(__file__).resolve().parent
    python = sys.executable
    # 统一参数透传给各子脚本, 保证路径和阈值一致
    common = [
        "--raw-dir",
        str(raw_dir),
        "--processed-dir",
        str(processed_dir),
        "--intermediate-dir",
        str(intermediate_dir),
        "--blur-threshold",
        str(blur_threshold),
        "--similarity-threshold",
        str(similarity_threshold),
    ]

    start = time.perf_counter()

    if not skip_generate:
        run_step([python, str(script_dir / "generate_mock_data.py"), *common, "--seed", str(seed)])
    else:
        print("Skipping mock data generation.")

    run_step([python, str(script_dir / "filter_blur.py"), *common])
    run_step([python, str(script_dir / "dedup_frames.py"), *common])

    metadata_cmd = [
        python,
        str(script_dir / "build_metadata.py"),
        *common,
        "--output-csv",
        str(output_csv),
    ]
    if plot:
        metadata_cmd.append("--plot")
    run_step(metadata_cmd)

    elapsed = time.perf_counter() - start
    blur_stats = load_json(intermediate_dir / "blur_stats.json")
    dedup_stats = load_json(intermediate_dir / "dedup_stats.json")

    print("\nPipeline finished")
    print(f"Total elapsed: {elapsed:.2f}s")
    print(f"Blur stage speed: {blur_stats['images_per_second']} images/sec")
    print(f"Dedup stage speed: {dedup_stats['images_per_second']} images/sec")
    print(f"Kept frames: {dedup_stats['kept']}")


def main() -> None:
    """命令行入口: 解析参数并一键运行完整 pipeline."""
    parser = argparse.ArgumentParser(description="Run the robot image cleaning pipeline.")
    add_common_args(parser)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=Path("metadata.csv"))
    args = parser.parse_args()

    run_pipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        intermediate_dir=args.intermediate_dir,
        blur_threshold=args.blur_threshold,
        similarity_threshold=args.similarity_threshold,
        skip_generate=args.skip_generate,
        seed=args.seed,
        plot=args.plot,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
