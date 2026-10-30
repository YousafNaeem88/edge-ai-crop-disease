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

The original capped run used one epoch and is retained for comparison. The
stronger recommended run uses all available training images, torchvision
pretrained initialization, five epochs, batch size 64, balanced sampling,
leaf-safe augmentation, image size 224, and seed 42.

| Metric | Value |
|---|---:|
| Test examples | 1,731 |
| Accuracy | 97.05% |
| Macro precision | 96.22% |
| Macro recall | 96.15% |
| Macro F1 | 96.20% |
| Expected calibration error (10 bins) | 13.65% |
| Median CPU latency | 15.49 ms |

The dynamic-quantized checkpoint reaches 96.94% accuracy and 96.01% macro-F1
with a 17.66 ms median CPU latency. Quantization is an experimental
compatibility path because eager-mode quantization is deprecated in newer
PyTorch releases.

These are curated-dataset results, not a field-diagnostic accuracy claim. The
run has not been validated on farmer-captured images or a physical edge
device. A high held-out score can still overestimate real-world performance
when images share collection conditions or visual backgrounds.

## Data limitations

The dataset metadata is preserved in `results/indian_multicrop_dataset.json`.
Before redistribution, verify the upstream dataset card's licensing terms.
The source covers the requested cotton, wheat, corn, and sugarcane groups but
does not contain mustard or the full vegetable scope from the proposal.
