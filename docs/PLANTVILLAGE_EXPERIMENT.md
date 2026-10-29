# PlantVillage experiment record

## Dataset

The experiment uses the public PlantVillage archive from the dataset authors'
Hugging Face mirror:

`https://huggingface.co/datasets/mohanty/PlantVillage`

The downloaded archive contains **54,306 images** across 38 labelled classes
covering 14 crop groups. The local audit found 54,305 valid image files; one
file was excluded during validation. The archive and extracted images are local
artifacts and are intentionally not committed to Git.

The original dataset paper is:

> Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using Deep Learning for
> Image-Based Plant Disease Detection. *Frontiers in Plant Science*, 7, 1419.
> https://doi.org/10.3389/fpls.2016.01419

## Reproducible commands

```powershell
python scripts/download_plantvillage.py
python -m edge_ai.audit --input data/raw --output results/plantvillage_audit.json
python scripts/train_plantvillage.py --epochs 1 --max-per-class 100 --device cpu
python -m edge_ai.evaluate --checkpoint artifacts/plantvillage_resnet18.pt --manifest data/plantvillage_manifest.csv --output results/plantvillage_resnet18.json
python -m edge_ai.evaluate --checkpoint artifacts/plantvillage_mobilenet_v3_small.pt --manifest data/plantvillage_manifest.csv --output results/plantvillage_mobilenet_v3_small.json
python -m edge_ai.quantize --checkpoint artifacts/plantvillage_mobilenet_v3_small.pt --output artifacts/plantvillage_mobilenet_v3_small_quantized.pt
```

The reported training run used at most 100 images per class, preserving all
38 classes. The full dataset remains available locally for a larger run.
This cap is a compute-control choice, not a claim that the source dataset is
small.

## Measured results

| Model | Test examples | Accuracy | Macro F1 | Parameters | Artifact size | Median CPU inference |
|---|---:|---:|---:|---:|---:|---:|
| ResNet18 | 570 | 9.65% | 0.0324 | 11,196,006 | 44.86 MB | 74.38 ms |
| MobileNetV3-Small | 570 | 71.58% | 0.6886 | 1,556,806 | 6.36 MB | 22.28 ms |

The MobileNetV3-Small result is the current deployed checkpoint at
`artifacts/model.pt` in the local working copy. A one-epoch, capped experiment
is not a final scientific claim; it is a reproducible baseline for subsequent
tuning and additional epochs.

Dynamic quantization reduced the MobileNetV3-Small checkpoint from 6,356,039
bytes to 4,486,603 bytes in this environment. Quantized accuracy must be
measured separately before treating it as the final model.

## Interpretation and limitations

- PlantVillage images are largely curated and may not represent field
  backgrounds, lighting, camera devices, or local varieties.
- The random class-balanced cap is useful for a practical CPU experiment but
  is not the official full-dataset split.
- Results are measured on this development computer and do not establish
  Raspberry Pi speed, power, thermal, or battery performance.
- The application output is a model prediction, not a confirmed diagnosis.
