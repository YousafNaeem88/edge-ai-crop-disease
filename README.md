# Edge-AI Crop Disease Detection

An offline-capable Python prototype for crop-disease image classification. The project provides:

- a Streamlit application for local prediction and SQLite history;
- a folder-per-class dataset preparation command with reproducible splits;
- PyTorch ResNet18 and MobileNetV3-Small training commands;
- evaluation, latency benchmarking, and post-training quantization utilities;
- tests that run without downloading datasets or model weights.

The application never calls an external inference API. A trained checkpoint is required for real predictions. Until one is supplied, the UI runs in clearly labelled **demo mode** using a deterministic image-colour heuristic; demo outputs are not model results.

## Quick start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
streamlit run app.py
```

Open the displayed local URL. The app starts safely without model artifacts and explains the next setup step.

## Dataset and training

Place a public folder-per-class dataset under `data/raw`, for example:

```text
data/raw/
  healthy/
    image_001.jpg
  tomato_blight/
    image_002.jpg
```

Then create a manifest:

```powershell
python -m edge_ai.audit --input data/raw --output results/dataset_audit.json
python -m edge_ai.data prepare --input data/raw --output data/manifest.csv --seed 42
```

Train a baseline or lightweight model:

```powershell
python -m edge_ai.train --manifest data/manifest.csv --model resnet18 --output artifacts/resnet18.pt
python -m edge_ai.train --manifest data/manifest.csv --model mobilenet_v3_small --output artifacts/mobilenet_v3_small.pt
```

Evaluate and benchmark:

```powershell
python -m edge_ai.evaluate --checkpoint artifacts/resnet18.pt --manifest data/manifest.csv --output results/resnet18.json
python -m edge_ai.quantize --checkpoint artifacts/resnet18.pt --output artifacts/resnet18_quantized.pt
python -m edge_ai.benchmark --checkpoint artifacts/resnet18.pt --image path/to/test.jpg --output results/latency.json
```

The evaluation JSON contains accuracy, macro precision/recall/F1, a per-class report,
confusion matrix, parameter count, checkpoint size, and repeatable CPU latency measurements.
The Streamlit **Results** page displays saved evaluation JSON files.

The training command supports `--epochs`, `--batch-size`, `--image-size`, `--seed`, and `--device`. Use `--no-pretrained` when working without internet access or cached torchvision weights.

The audit command records valid/invalid files, class counts, common image dimensions,
and explicit limitations before model training. Resource profiles are documented in
`configs/resource_profiles.yaml`; they must not be described as physical-device
measurements.

For deployment and evidence collection, see `docs/PROJECT_OPERATIONS.md`. A `Dockerfile`
is included for a self-contained local deployment.

## Offline verification

After installing dependencies and placing a checkpoint locally, disable network access and run:

```powershell
streamlit run app.py
```

Upload an image, verify a prediction is returned, and confirm the record appears on the History page. The inference service loads only local checkpoint/configuration files.

## Project limitations

Software resource profiles are analytical constraints, not physical-device measurements. This project does not claim Raspberry Pi performance, energy consumption, thermal behaviour, or field-level diagnostic accuracy. Predictions are decision-support outputs, not agricultural diagnoses.
