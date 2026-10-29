"""Download and extract the public PlantVillage archive.

The archive is intentionally not committed to Git. It contains 54,306 images
covering 14 crops and 26 disease/health classes.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/downloads/plantvillage.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.archive.exists():
        print(f"Downloading PlantVillage from {args.url}")
        urllib.request.urlretrieve(args.url, args.archive)
    print(f"Extracting {args.archive} to a temporary directory")
    temporary = args.output.parent / "plantvillage_extract"
    if temporary.exists():
        shutil.rmtree(temporary)
    with zipfile.ZipFile(args.archive) as archive:
        archive.extractall(temporary)
    candidates = list(temporary.rglob("color"))
    source = next((path for path in candidates if path.is_dir()), temporary)
    for item in source.iterdir():
        destination = args.output / item.name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(item), destination)
    shutil.rmtree(temporary)
    print(f"Dataset extracted to {args.output}")


if __name__ == "__main__":
    main()
