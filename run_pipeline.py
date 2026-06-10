#!/usr/bin/env python3
"""Run the full robot image cleaning pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from pipeline_common import add_common_args, load_json


def run_step(command: list[str]) -> None:
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
    script_dir = Path(__file__).resolve().parent
    python = sys.executable
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
