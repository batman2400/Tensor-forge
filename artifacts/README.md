# artifacts/

Frozen, deployable model files produced by `ml/` (classical models, label maps, ONNX encoder,
tokenizer, calibration and thresholds) plus `manifest.json`.

- `manifest.json` records `model_version` and a SHA-256 for every artifact. The same
  `model_version` is returned by `/health`, `/predict`, `/predict/batch` and the job endpoints.
- Artifacts are baked into the Docker image; the service never downloads anything at runtime.
- Everything here is gitignored except `README.md` and `manifest.json`. Final artifacts are
  added to the repo (or attached to a GitHub Release / Git LFS if a file exceeds 100 MB) when the
  model is frozen. See the project plan for the decision.
