"""Command-line interface for public evaluation utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .assignment import assignment_metrics
from .metrics import aligned_rmsd, raw_rmsd, summarize
from .preprocessing import describe_images


def load_array(path: str) -> np.ndarray:
    return np.load(Path(path), mmap_mode="r")


def write_summary(summary: dict, output: str | None) -> None:
    text = json.dumps(summary, indent=2, sort_keys=True)
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
    print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cryoem-coords")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Compute raw and aligned paired RMSD.")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--targets", required=True)
    evaluate.add_argument("--output")

    assignment = subparsers.add_parser("assignment", help="Compute target-PCA paired-target rank diagnostics.")
    assignment.add_argument("--predictions", required=True)
    assignment.add_argument("--targets", required=True)
    assignment.add_argument("--components", type=int, default=10)
    assignment.add_argument("--chunk-size", type=int, default=256)
    assignment.add_argument("--seed", type=int, default=42)
    assignment.add_argument("--output")

    inspect_images = subparsers.add_parser("inspect-images", help="Describe a saved image array without changing it.")
    inspect_images.add_argument("--images", required=True)
    inspect_images.add_argument("--sample-size", type=int, default=1024)
    inspect_images.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "evaluate":
        predictions = load_array(args.predictions)
        targets = load_array(args.targets)
        summary = {
            "raw_rmsd": summarize(raw_rmsd(predictions, targets)),
            "aligned_rmsd": summarize(aligned_rmsd(predictions, targets)),
        }
        write_summary(summary, args.output)
    elif args.command == "assignment":
        summary = assignment_metrics(
            load_array(args.predictions),
            load_array(args.targets),
            components=args.components,
            chunk_size=args.chunk_size,
            seed=args.seed,
        )
        write_summary(summary, args.output)
    elif args.command == "inspect-images":
        write_summary(describe_images(load_array(args.images), sample_size=args.sample_size), args.output)


if __name__ == "__main__":
    main()
