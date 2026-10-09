# artifacts/

Frozen, deployable model files produced by `ml/` (classical models, label maps, ONNX encoder,
tokenizer, calibration and thresholds) plus `manifest.json`.

- `manifest.json` records `model_version` and a SHA-256 for every artifact. The same
  `model_version` is returned by `/health`, `/predict`, `/predict/batch` and the job endpoints.
- Artifacts are baked into the Docker image; the service never downloads anything at runtime.
- `classical.joblib`, `tokenizer.json`, `encoder_meta.json`, `fusion.json`, and `manifest.json` are tracked in git. `encoder.int8.onnx` is tracked with Git LFS because it is over GitHub's 100 MB file limit.
