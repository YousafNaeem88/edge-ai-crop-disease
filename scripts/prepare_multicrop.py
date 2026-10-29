from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SOURCE = "rohitashva/indian_crop_diseases"
LICENSE = "Dataset license/terms should be verified from the Hugging Face dataset card before redistribution."


def crop_from_label(label: str) -> str:
    return label.split("_", 1)[0].strip().lower()


def prepare(input_dir: Path, output: Path, metadata_path: Path) -> None:
    rows: list[dict[str, str]] = []
    for split_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        split_name = "validation" if split_dir.name in {"valid", "validation"} else split_dir.name
        if split_name not in {"train", "validation", "test"}:
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            label = class_dir.name
            crop = crop_from_label(label)
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() not in EXTENSIONS:
                    continue
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except (UnidentifiedImageError, OSError):
                    continue
                rows.append(
                    {
                        "path": str(image_path.resolve()),
                        "label": label,
                        "crop": crop,
                        "disease": label.split("_", 1)[1] if "_" in label else label,
                        "split": split_name,
                        "source_dataset": SOURCE,
                        "license": LICENSE,
                    }
                )
    if not rows:
        raise ValueError(f"No valid images found under {input_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "source_dataset": SOURCE,
        "license_note": LICENSE,
        "input_dir": str(input_dir.resolve()),
        "images": len(rows),
        "classes": sorted({row["label"] for row in rows}),
        "crops": sorted({row["crop"] for row in rows}),
        "split_counts": {
            split: sum(1 for row in rows if row["split"] == split)
            for split in ("train", "validation", "test")
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} images across {len(metadata['classes'])} classes to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Indian multi-crop dataset")
    parser.add_argument("--input", type=Path, default=Path("data/raw/indian_crop_diseases/dataset"))
    parser.add_argument("--output", type=Path, default=Path("data/indian_multicrop_manifest.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("results/indian_multicrop_dataset.json"))
    args = parser.parse_args()
    prepare(args.input, args.output, args.metadata)


if __name__ == "__main__":
    main()
