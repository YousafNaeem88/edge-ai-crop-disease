from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_ai.train import train


def write_crop_manifest(source: Path, crop: str, output: Path) -> int:
    with source.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("crop") == crop]
    if not rows:
        raise ValueError(f"No rows found for crop: {crop}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train crop-specific specialist models")
    parser.add_argument("--manifest", type=Path, default=Path("data/indian_multicrop_manifest.csv"))
    parser.add_argument("--crop", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    args.output = args.output or Path(
        f"artifacts/indian_{args.crop}_mobilenet_v3_small_specialist.pt"
    )
    specialist_manifest = Path(f"data/indian_{args.crop}_manifest.csv")
    count = write_crop_manifest(args.manifest, args.crop, specialist_manifest)
    args.manifest = specialist_manifest
    args.model = "mobilenet_v3_small"
    args.balanced_sampling = True
    print(f"Training {args.crop} specialist on {count} images")
    train(args)


if __name__ == "__main__":
    main()
