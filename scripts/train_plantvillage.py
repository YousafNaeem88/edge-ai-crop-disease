"""Convenience entry point for a reproducible PlantVillage experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from edge_ai.data import build_manifest
from edge_ai.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-per-class", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest = Path("data/plantvillage_manifest.csv")
    build_manifest(args.data, manifest, seed=42, max_per_class=args.max_per_class)
    for model_name in ("resnet18", "mobilenet_v3_small"):
        output = Path("artifacts") / f"plantvillage_{model_name}.pt"
        train_args = argparse.Namespace(
            manifest=manifest,
            model=model_name,
            output=output,
            epochs=args.epochs,
            batch_size=32,
            image_size=224,
            learning_rate=1e-3,
            seed=42,
            device=args.device,
            no_pretrained=False,
        )
        train(train_args)
