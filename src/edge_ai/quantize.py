from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Attempt dynamic quantization of a checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        import torch
        from torchvision import models
    except ImportError as exc:
        raise SystemExit("Install torch and torchvision before quantization") from exc
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = models.resnet18(weights=None) if architecture == "resnet18" else models.mobilenet_v3_small(weights=None)
    if architecture == "resnet18":
        model.fc = torch.nn.Linear(model.fc.in_features, len(checkpoint["labels"]))
    else:
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(checkpoint["labels"]))
    model.load_state_dict(checkpoint["state_dict"])
    quantized = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**checkpoint, "state_dict": quantized.state_dict(), "quantized": True}, args.output)
    print(f"Quantized checkpoint written to {args.output}")


if __name__ == "__main__":
    main()

