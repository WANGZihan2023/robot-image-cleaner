#!/usr/bin/env python3
"""汇总 metadata: 合并 blur 与 dedup 两阶段结果, 输出 metadata.csv.

合并规则:
  1. 以 blur_results.json 为全量基准 (每帧一行)
  2. blur 已剔除 -> reason 保持 blur
  3. blur 通过但 dedup 判重复 -> 覆盖为 duplicate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pipeline_common import add_common_args, load_json


def build_metadata(
    intermediate_dir: Path,
    output_csv: Path,
    plot: bool = False,
    reports_dir: Path | None = None,
) -> pd.DataFrame:
    """合并 blur 与 dedup 结果, 输出 metadata.csv, 可选生成统计图."""
    blur_path = intermediate_dir / "blur_results.json"
    dedup_path = intermediate_dir / "dedup_results.json"
    if not blur_path.exists():
        raise FileNotFoundError(f"Missing blur results: {blur_path}")
    if not dedup_path.exists():
        raise FileNotFoundError(f"Missing dedup results: {dedup_path}")

    blur_results = load_json(blur_path)
    dedup_results = load_json(dedup_path)

    rows: list[dict] = []
    for episode_id in sorted(blur_results.keys()):
        blur_records = {record["frame_id"]: record for record in blur_results[episode_id]}
        dedup_records = {record["frame_id"]: record for record in dedup_results.get(episode_id, [])}

        for frame_id in sorted(blur_records.keys(), key=lambda value: int(value.split("_")[1])):
            blur_record = blur_records[frame_id]
            dedup_record = dedup_records.get(frame_id, blur_record)

            is_valid = blur_record["is_valid"]
            reason = blur_record["reason"]
            # dedup 阶段才会产生 duplicate 标记
            if blur_record["is_valid"] and not dedup_record.get("is_valid", True):
                is_valid = False
                reason = dedup_record.get("reason", "duplicate")

            rows.append(
                {
                    "episode_id": episode_id,
                    "frame_id": frame_id,
                    "file_path": blur_record["file_path"],
                    "blur_score": blur_record["blur_score"],
                    "is_valid": is_valid,
                    "reason": reason,
                }
            )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    total = len(df)
    kept = int(df["is_valid"].sum())
    blur_rejected = int((df["reason"] == "blur").sum())
    duplicate_rejected = int((df["reason"] == "duplicate").sum())
    reject_rate = round((total - kept) / total * 100, 2) if total else 0.0

    print("Metadata build complete")
    print(f"Original frames: {total}")
    print(f"Kept frames: {kept}")
    print(f"Rejected - blur: {blur_rejected}")
    print(f"Rejected - duplicate: {duplicate_rejected}")
    print(f"Reject rate: {reject_rate}%")
    print(f"Saved metadata to {output_csv}")

    if plot:
        reports_dir = reports_dir or Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        plot_path = reports_dir / "cleaning_summary.png"

        labels = ["Kept", "Blur", "Duplicate"]
        values = [kept, blur_rejected, duplicate_rejected]
        colors = ["#2ecc71", "#e74c3c", "#f39c12"]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(["Before", "After"], [total, kept], color=["#3498db", "#2ecc71"])
        axes[0].set_title("Frames Before vs After Cleaning")
        axes[0].set_ylabel("Count")

        axes[1].bar(labels, values, color=colors)
        axes[1].set_title("Rejection Reasons")
        axes[1].set_ylabel("Count")

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved summary plot to {plot_path}")

    return df


def main() -> None:
    """命令行入口: 解析参数并生成 metadata.csv."""
    parser = argparse.ArgumentParser(description="Build metadata.csv from pipeline results.")
    add_common_args(parser)
    parser.add_argument("--output-csv", type=Path, default=Path("metadata.csv"))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    build_metadata(args.intermediate_dir, args.output_csv, plot=args.plot, reports_dir=args.reports_dir)


if __name__ == "__main__":
    main()
