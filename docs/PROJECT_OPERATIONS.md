# Project operations and final-year-project evidence

## Reproducible experiment sequence

1. Record the machine, Python, framework, and dataset versions.
2. Audit the public dataset and record its license, class counts, unreadable files, and duplicates.
3. Generate `data/manifest.csv` with a fixed seed. The splitter keeps classes separate and creates train, validation, and test partitions.
4. Train ResNet18 and MobileNetV3-Small with the same manifest and record the command output.
5. Evaluate both checkpoints on the untouched test partition.
6. Attempt post-training dynamic quantization and record whether the runtime supports the exported artifact.
7. Benchmark local latency with repeated batch-one inference.
8. Run the application with network access disabled and retain screenshots or a test record.

## Evidence to retain

- dataset feasibility sheet and license URL;
- manifest and class distribution summary;
- training commands, seeds, and checkpoint metadata;
- `results/*.json` evaluation and benchmark files;
- screenshots of Predict, History, Results, and offline verification;
- limitations section distinguishing measured, simulated, and unavailable hardware metrics.

## Deployment

For a local machine:

```powershell
pip install .
$env:PYTHONPATH="src"
streamlit run app.py
```

For a container:

```powershell
docker build -t edge-ai-crop-disease .
docker run --rm -p 8501:8501 edge-ai-crop-disease
```

Mount `data`, `artifacts`, and `results` when deploying a trained model. No external API or database server is required.
