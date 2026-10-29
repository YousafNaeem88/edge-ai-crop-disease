from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def train(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    with Path(args.manifest).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = sorted({row["label"] for row in rows})
    label_to_index = {label: index for index, label in enumerate(labels)}
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])
    samples = [(row["path"], label_to_index[row["label"]]) for row in rows if row["split"] == "train"]
    validation_samples = [
        (row["path"], label_to_index[row["label"]]) for row in rows if row["split"] == "validation"
    ]
    class ManifestDataset(Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            path, target = self.items[index]
            with Image.open(path) as image:
                image = image.convert("RGB")
            return transform(image), target

    dataset = ManifestDataset(samples)
    validation_dataset = ManifestDataset(validation_samples)
    if args.model == "resnet18":
        model = models.resnet18(weights=None if args.no_pretrained else models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(labels))
    else:
        model = models.mobilenet_v3_small(weights=None if args.no_pretrained else models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(labels))
    model.to(device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), targets.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
        model.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for images, targets in validation_loader:
                validation_loss += float(criterion(model(images.to(device)), targets.to(device)))
        validation_loss /= max(len(validation_loader), 1)
        print(
            f"epoch={epoch + 1} train_loss={total / max(len(loader), 1):.4f} "
            f"validation_loss={validation_loss:.4f}"
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": args.model,
            "labels": labels,
            "state_dict": best_state or model.cpu().state_dict(),
            "image_size": args.image_size,
            "seed": args.seed,
            "best_validation_loss": best_loss if best_state is not None else None,
        },
        args.output,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a crop-disease classifier")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", choices=["resnet18", "mobilenet_v3_small"], default="resnet18")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
