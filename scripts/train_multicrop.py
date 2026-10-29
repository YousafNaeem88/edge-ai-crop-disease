from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_ai.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the multi-crop Edge-AI classifier")
    parser.add_argument("--manifest", type=Path, default=Path("data/indian_multicrop_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/indian_multicrop_mobilenet_v3_small.pt"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=100)
    args = parser.parse_args()
    if args.max_per_class:
        with args.manifest.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rng = random.Random(args.seed)
        selected: list[dict[str, str]] = []
        for label in sorted({row["label"] for row in rows}):
            for split in ("train", "validation", "test"):
                candidates = [row for row in rows if row["label"] == label and row["split"] == split]
                rng.shuffle(candidates)
                selected.extend(candidates[: args.max_per_class])
        capped = args.manifest.with_name(args.manifest.stem + "_capped.csv")
        with capped.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
        args.manifest = capped
    args.model = "mobilenet_v3_small"
    train(args)


if __name__ == "__main__":
    main()
