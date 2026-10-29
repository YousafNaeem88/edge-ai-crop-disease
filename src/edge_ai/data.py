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
    random.Random(seed).shuffle(records)
    counts = {label: sum(1 for _, item in records if item == label) for label in sorted({x[1] for x in records})}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "split"])
        writer.writeheader()
        for index, (path, label) in enumerate(records):
            ratio = index % 20
            split = "test" if ratio < 3 else "validation" if ratio < 6 else "train"
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

