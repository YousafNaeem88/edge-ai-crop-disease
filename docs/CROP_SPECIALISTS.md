# Crop-specific specialist models

The local application supports a unified multi-crop model and separate
MobileNetV3-Small specialists. Select a model from the sidebar when the crop
is known. Specialist models use a narrower label space and can be preferable
when the crop has already been identified.

## Measured results

| Model | Test examples | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Unified multi-crop | 1,731 | 97.05% | 96.20% |
| Corn specialist | 421 | 94.06% | 92.73% |
| Cotton specialist | 324 | 99.69% | 99.69% |
| Jute specialist | 93 | 79.57% | 79.43% |
| Rice specialist | 596 | 100.00% | 100.00% |
| Sugarcane specialist | 255 | 94.12% | 94.00% |
| Wheat specialist | 42 | 95.24% | 95.56% |

These metrics are not directly interchangeable: each specialist is evaluated
only on its crop's test split, while the unified model is evaluated across all
crops. The small wheat and jute test sets make their estimates less stable.
Use these as reproducible project evidence, not field-diagnostic guarantees.

## Local training

```powershell
python scripts/train_crop_specialists.py --crop corn --epochs 4
python scripts/train_crop_specialists.py --crop cotton --epochs 4
python scripts/train_crop_specialists.py --crop jute --epochs 4
python scripts/train_crop_specialists.py --crop rice --epochs 4
python scripts/train_crop_specialists.py --crop sugarcane --epochs 4
python scripts/train_crop_specialists.py --crop wheat --epochs 4
```

The generated checkpoints are intentionally ignored by Git and can be
published as release assets.
