from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError


class PredictionError(ValueError):
    """Raised when an input cannot be safely processed."""


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    model_name: str
    latency_ms: float
    is_demo: bool = False


class InferenceService:
    def __init__(self, model=None, labels: list[str] | None = None, model_name: str = "demo") -> None:
        self.model = model
        self.labels = labels or ["healthy", "possible_leaf_disease"]
        self.model_name = model_name
        self.is_demo = model is None
        self._transform = None

    @classmethod
    def demo(cls) -> "InferenceService":
        return cls()

    @classmethod
    def from_checkpoint(cls, path: Path) -> "InferenceService":
        try:
            import torch
            from torchvision import models, transforms
        except ImportError as exc:
            raise PredictionError("PyTorch and torchvision are required to load a checkpoint") from exc
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        labels = checkpoint.get("labels")
        architecture = checkpoint.get("architecture", "resnet18")
        factory = models.resnet18 if architecture == "resnet18" else models.mobilenet_v3_small
        model = factory(weights=None)
        if architecture == "resnet18":
            model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
        else:
            model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(labels))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        service = cls(model, labels, architecture)
        service._transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        return service

    def _image(self, payload: bytes) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(payload)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise PredictionError("The uploaded file is not a readable image") from exc
        if image.width < 16 or image.height < 16:
            raise PredictionError("Image is too small; upload an image at least 16×16 pixels")
        return image

    def predict_bytes(self, payload: bytes) -> Prediction:
        image = self._image(payload)
        started = time.perf_counter()
        if self.is_demo:
            pixels = np.asarray(image, dtype=np.float32).mean(axis=(0, 1))
            green_ratio = float(pixels[1] / max(pixels.mean(), 1))
            label = self.labels[0] if green_ratio >= 1.0 else self.labels[1]
            confidence = min(0.95, max(0.55, abs(green_ratio - 0.8)))
        else:
            import torch
            tensor = self._transform(image).unsqueeze(0)
            with torch.inference_mode():
                probabilities = torch.softmax(self.model(tensor), dim=1)[0]
            index = int(torch.argmax(probabilities))
            label, confidence = self.labels[index], float(probabilities[index])
        return Prediction(label, confidence, self.model_name, (time.perf_counter() - started) * 1000, self.is_demo)

