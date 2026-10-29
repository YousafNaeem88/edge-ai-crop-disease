# Edge-AI Crop Disease Detection

An offline-capable Python prototype for crop-disease image classification. The project provides:

- a Streamlit application for local prediction and SQLite history;
- a folder-per-class dataset preparation command with reproducible splits;
- PyTorch ResNet18 and MobileNetV3-Small training commands;
- evaluation, latency benchmarking, and post-training quantization utilities;
- tests that run without downloading datasets or model weights.

The application never calls an external inference API. A trained checkpoint is required for real predictions. Until one is supplied, the UI runs in clearly labelled **demo mode** using a deterministic image-colour heuristic; demo outputs are not model results.

## Download the trained checkpoint

A real MobileNetV3-Small checkpoint trained on all 38 PlantVillage classes is
available in the [v0.1.0 release](https://github.com/YousafNaeem88/edge-ai-crop-disease/releases/tag/v0.1.0).
Download `model.pt` into `artifacts/model.pt`, then launch the app. The release
also contains an experimental quantized artifact.

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
Invoke-WebRequest `
  -Uri https://github.com/YousafNaeem88/edge-ai-crop-disease/releases/download/v0.1.0/model.pt `
  -OutFile artifacts/model.pt
streamlit run app.py
```

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

To download the complete public PlantVillage archive (54,306 images, 14 crops,
and 38 labelled classes) and run a reproducible two-model experiment:

```powershell
python scripts/download_plantvillage.py
python -m edge_ai.audit --input data/raw --output results/plantvillage_audit.json
python scripts/train_plantvillage.py --epochs 2 --max-per-class 1000
```

`--max-per-class` is a compute control only; the full archive remains available
locally. Set it to a larger value or remove the limit when hardware permits.
The checked-in experiment record and measured baseline results are in
`docs/PLANTVILLAGE_EXPERIMENT.md`.

### Indian multi-crop baseline

The repository also includes a reproducible acquisition and training workflow
for an Indian multi-crop source containing cotton, wheat, corn, sugarcane,
rice, and jute:

```powershell
# The archive is downloaded separately; do not commit it.
Invoke-WebRequest `
  -Uri https://huggingface.co/datasets/rohitashva/indian_crop_diseases/resolve/main/dataset.zip `
  -OutFile data/downloads/indian_crop_diseases.zip
Expand-Archive data/downloads/indian_crop_diseases.zip -DestinationPath data/raw/indian_crop_diseases -Force
python scripts/prepare_multicrop.py
python scripts/train_multicrop.py --epochs 1 --max-per-class 100
$env:PYTHONPATH = "src"
python -m edge_ai.evaluate `
  --checkpoint artifacts/indian_multicrop_mobilenet_v3_small.pt `
  --manifest data/indian_multicrop_manifest_capped.csv `
  --output results/indian_multicrop_mobilenet.json
```

The current capped baseline reaches 70.31% accuracy, 68.68% macro-F1, and
15.57 ms median CPU latency on 1,472 held-out examples. Full details,
per-crop metrics, provenance, and limitations are in
`docs/INDIAN_MULTICROP_EXPERIMENT.md`. Mustard is explicitly not claimed by
this model because the source contains no mustard classes.

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
