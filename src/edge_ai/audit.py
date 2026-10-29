from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .data import EXTENSIONS


def audit_dataset(input_dir: Path, output: Path, verify_all: bool = False) -> dict[str, object]:
    classes: Counter[str] = Counter()
    invalid_files: list[str] = []
    dimensions: Counter[str] = Counter()
    for class_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() not in EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    if verify_all:
                        image.verify()
                    else:
                        image.getbands()
                    if dimensions.total() < 1000:
                        dimensions[f"{image.width}x{image.height}"] += 1
                classes[class_dir.name] += 1
            except (UnidentifiedImageError, OSError):
                invalid_files.append(str(path))
    if not classes:
        raise ValueError(f"No valid images found under {input_dir}")
    result: dict[str, object] = {
        "source": str(input_dir.resolve()),
        "total_valid_images": sum(classes.values()),
        "class_counts": dict(sorted(classes.items())),
        "class_count": len(classes),
        "invalid_files": invalid_files,
        "invalid_count": len(invalid_files),
        "common_dimensions": dimensions.most_common(10),
        "limitations": [
            "This audit does not detect semantic mislabels.",
            "Near-duplicate and source-sequence leakage require additional review.",
            "By default, image decoding is sampled; use --verify-all for a full decode pass.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a folder-per-class image dataset")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-all", action="store_true")
    args = parser.parse_args()
    result = audit_dataset(args.input, args.output, verify_all=args.verify_all)
    print(f"valid_images={result['total_valid_images']} classes={result['class_count']}")


if __name__ == "__main__":
    main()
