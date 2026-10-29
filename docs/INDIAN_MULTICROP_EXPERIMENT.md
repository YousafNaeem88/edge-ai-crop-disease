# Indian multi-crop experiment

This experiment uses the public Hugging Face dataset
`rohitashva/indian_crop_diseases`. The downloaded archive contains fixed
train/validation/test folders and 17,168 valid images across 23 classes:
corn, cotton, jute, rice, sugarcane, and wheat. Mustard is not present in this
source and is not represented by fabricated or relabelled data.

## Reproduce

```powershell
python scripts/prepare_multicrop.py
python scripts/train_multicrop.py --epochs 1 --max-per-class 100
$env:PYTHONPATH = "src"
python -m edge_ai.evaluate `
  --checkpoint artifacts/indian_multicrop_mobilenet_v3_small.pt `
  --manifest data/indian_multicrop_manifest_capped.csv `
  --output results/indian_multicrop_mobilenet.json
```

The cap is applied independently to each class and split so the experiment
remains practical on a CPU. Increase it or remove it for a larger run.

## Measured baseline

The current reproducible capped run uses MobileNetV3-Small with torchvision
pretrained initialization, one training epoch, batch size 32, image size 224,
and seed 42.

| Metric | Value |
|---|---:|
| Test examples | 1,472 |
| Accuracy | 70.31% |
| Macro precision | 76.22% |
| Macro recall | 69.26% |
| Macro F1 | 68.68% |
| Median CPU latency | 15.57 ms |

These are a baseline experiment, not a field-diagnostic accuracy claim.
Images are curated dataset examples, and the run has not been validated on
farmer-captured images or a physical edge device.

## Data limitations

The dataset metadata is preserved in `results/indian_multicrop_dataset.json`.
Before redistribution, verify the upstream dataset card's licensing terms.
The source covers the requested cotton, wheat, corn, and sugarcane groups but
does not contain mustard or the full vegetable scope from the proposal.
