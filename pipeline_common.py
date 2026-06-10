"""Shared helpers for the robot image cleaning pipeline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

FRAME_PATTERN = re.compile(r"frame_(\d+)", re.IGNORECASE)
OUTPUT_SIZE = (224, 224)


def add_common_args(parser: argparse.ArgumentParser) -> None:
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
    match = FRAME_PATTERN.search(frame_id)
    if not match:
        raise ValueError(f"Invalid frame id: {frame_id}")
    return int(match.group(1))


def frame_sort_key(frame_id: str) -> int:
    return parse_frame_index(frame_id)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def iter_episode_images(raw_dir: Path) -> list[tuple[str, str, Path]]:
    records: list[tuple[str, str, Path]] = []
    for image_path in sorted(raw_dir.rglob("*.jpg")):
        episode_id = image_path.parent.name
        frame_id = image_path.stem
        records.append((episode_id, frame_id, image_path))
    return records


def group_by_episode(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["episode_id"], []).append(record)
    for episode_records in grouped.values():
        episode_records.sort(key=lambda item: frame_sort_key(item["frame_id"]))
    return grouped
