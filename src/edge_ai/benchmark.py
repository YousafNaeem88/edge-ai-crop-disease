from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inference import InferenceService


def benchmark(checkpoint: Path, image: Path, repeats: int) -> dict[str, float | int | str]:
    payload = image.read_bytes()
    service = InferenceService.from_checkpoint(checkpoint)
    samples: list[float] = []
    for _ in range(repeats):
        result = service.predict_bytes(payload)
        samples.append(result.latency_ms)
    ordered = sorted(samples)
    return {
        "checkpoint": str(checkpoint),
        "image": str(image),
        "repeats": repeats,
        "median_latency_ms": ordered[len(ordered) // 2],
        "mean_latency_ms": sum(samples) / len(samples),
        "min_latency_ms": min(samples),
        "max_latency_ms": max(samples),
        "measurement": "Local CPU inference including preprocessing; batch size 1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local inference latency")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 3:
        raise SystemExit("--repeats must be at least 3")
    result = benchmark(args.checkpoint, args.image, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"median_latency_ms={result['median_latency_ms']:.2f}")


if __name__ == "__main__":
    main()
