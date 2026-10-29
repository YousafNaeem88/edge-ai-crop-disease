from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, UnidentifiedImageError

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def build_manifest(input_dir: Path, output: Path, seed: int = 42) -> None:
    records: list[tuple[str, str]] = []
    for class_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() not in EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError):
                continue
            records.append((str(path.resolve()), class_dir.name))
    if not records:
        raise ValueError(f"No valid images found under {input_dir}")
    rng = random.Random(seed)
    grouped: dict[str, list[str]] = {}
    for path, label in records:
        grouped.setdefault(label, []).append(path)
    records = []
    for label in sorted(grouped):
        examples = grouped[label]
        rng.shuffle(examples)
        total = len(examples)
        test_count = max(1, round(total * 0.15)) if total >= 3 else 0
        validation_count = max(1, round(total * 0.15)) if total >= 4 else 0
        for index, path in enumerate(examples):
            if index < test_count:
                split = "test"
            elif index < test_count + validation_count:
                split = "validation"
            else:
                split = "train"
            records.append((path, label, split))
    counts = {label: sum(1 for _, item, _ in records if item == label) for label in sorted({x[1] for x in records})}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "split"])
        writer.writeheader()
        for path, label, split in records:
            writer.writerow({"path": path, "label": label, "split": split})
    print(f"Wrote {len(records)} images across {len(counts)} classes to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducible image manifest")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_manifest(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()
