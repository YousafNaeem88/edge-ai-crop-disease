from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


def evaluate_checkpoint(checkpoint_path: Path, manifest_path: Path, output_path: Path) -> dict[str, Any]:
    import torch
    from PIL import Image
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from torchvision import models, transforms

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = checkpoint["labels"]
    architecture = checkpoint["architecture"]
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
    elif architecture == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(labels))
    else:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    if checkpoint.get("quantized"):
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    image_size = int(checkpoint.get("image_size", 224))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    label_to_index = {label: index for index, label in enumerate(labels)}
    with manifest_path.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "test"]
    if not rows:
        raise ValueError("The manifest contains no test examples")

    actual: list[int] = []
    predicted: list[int] = []
    confidences: list[float] = []
    latencies: list[float] = []
    with torch.inference_mode():
        for row in rows:
            with Image.open(row["path"]) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0)
            started = time.perf_counter()
            output = model(tensor)
            latencies.append((time.perf_counter() - started) * 1000)
            actual.append(label_to_index[row["label"]])
            probabilities = torch.softmax(output, dim=1)[0]
            predicted.append(int(probabilities.argmax().item()))
            confidences.append(float(probabilities.max().item()))

    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "architecture": architecture,
        "examples": len(rows),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=list(range(len(labels)))).tolist(),
        "labels": labels,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "artifact_size_bytes": checkpoint_path.stat().st_size,
        "latency_ms_median": float(sorted(latencies)[len(latencies) // 2]),
        "latency_ms_mean": float(sum(latencies) / len(latencies)),
        "measurement": "CPU inference only; preprocessing excluded; batch size 1",
    }
    bins = 10
    calibration_error = 0.0
    for lower in [index / bins for index in range(bins)]:
        upper = lower + (1 / bins)
        selected = [
            index
            for index, confidence in enumerate(confidences)
            if lower <= confidence < upper or (upper == 1.0 and confidence <= upper)
        ]
        if selected:
            accuracy = sum(actual[index] == predicted[index] for index in selected) / len(selected)
            mean_confidence = sum(confidences[index] for index in selected) / len(selected)
            calibration_error += len(selected) / len(confidences) * abs(accuracy - mean_confidence)
    result["expected_calibration_error_10bin"] = float(calibration_error)
    if "crop" in rows[0]:
        per_crop: dict[str, dict[str, Any]] = {}
        for crop in sorted({row["crop"] for row in rows}):
            indexes = [index for index, row in enumerate(rows) if row["crop"] == crop]
            crop_actual = [actual[index] for index in indexes]
            crop_predicted = [predicted[index] for index in indexes]
            crop_labels = sorted({value for value in crop_actual})
            per_crop[crop] = {
                "examples": len(indexes),
                "accuracy": float(accuracy_score(crop_actual, crop_predicted)),
                "macro_f1": float(
                    classification_report(
                        crop_actual,
                        crop_predicted,
                        labels=crop_labels,
                        target_names=[labels[index] for index in crop_labels],
                        output_dict=True,
                        zero_division=0,
                    )["macro avg"]["f1-score"]
                ),
            }
        result["per_crop"] = per_crop
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a local crop-disease checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_checkpoint(args.checkpoint, args.manifest, args.output)
    print(
        f"accuracy={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f} "
        f"median_latency_ms={result['latency_ms_median']:.2f}"
    )


if __name__ == "__main__":
    main()
