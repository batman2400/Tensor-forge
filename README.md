# TensorForge 2.0 - Phase 2 MVP: Support Ticket Classification and Routing

Team **Fade**. A self-trained classifier for the fictional RideEat ride-hailing and food-delivery
platform. No LLM is called at inference. For each support ticket the service predicts:

- `category` (11 classes) and the routed `team` (the fixed mapping from the organizers' spec)
- `secondary_category` (a second issue, or `null`)
- `is_urgent`
- a calibrated `confidence` for the primary category

`needs_human_review` is `true` when that confidence is below **0.5**.

The service follows the organizers' OpenAPI contract in [`spec/`](spec/): `POST /predict`,
`POST /predict/batch` (1-100 tickets), async `POST /batch/jobs` (up to 5,000), job polling,
results, `DELETE`, and a public `GET /health`. Every endpoint except `/health` requires the API key,
sent as `X-API-Key` or `Authorization: Bearer`.

## Live service

| | |
| --- | --- |
| Base URL | https://tensorforge-fade.southindia.cloudapp.azure.com |
| Health | https://tensorforge-fade.southindia.cloudapp.azure.com/health |
| Demo | https://tensorforge-fade.southindia.cloudapp.azure.com/demo/ |
| Model | `v1.0.0-38ecb9bd` (fused TF-IDF + int8 encoder) |
| Image | `sha256:e18b0c716e0240c4dc86f9a04703380e9a82cd2a604856e4d30d3e21da88a11c` |

The demo asks the visitor to paste the API key. The key is kept in that tab's session storage and is not in the page source.

Held-out score, fit on the 4,000 training tickets and scored on the 800 validation tickets: fusion macro-F1 **0.802** (classical `svc_word_char` alone was 0.656). Quote this figure. A score computed on the full 4,800 tickets after the final fit is not a held-out result. Details are in [`ml/reports/fusion.md`](ml/reports/fusion.md) and [`ml/reports/g1.md`](ml/reports/g1.md).

## Run the container

The container listens on `0.0.0.0:8000`, reads `API_KEY` from the environment, and does not need a network at runtime. Model files are baked into the image.

```bash
docker run -p 8000:8000 -e API_KEY=<key> tensorforge:verify
```

`GET /health` returns 200 once the model has loaded (a few seconds, well under the 120 second limit). On this machine the same image also starts with `--network none`.

Build it from this repo only when `artifacts/` already contains the frozen files listed below. `encoder.int8.onnx` is 118 MB, over GitHub's 100 MB file limit, so the weight files are not in git. `manifest.json` and `fusion.json` are.

```bash
docker build -t tensorforge:verify .
```

Files the build copies from `artifacts/`:

| File | In git |
| --- | --- |
| `manifest.json` | yes |
| `fusion.json` | yes |
| `classical.joblib` | no |
| `tokenizer.json` | no |
| `encoder_meta.json` | no |
| `encoder.int8.onnx` | no (118 MB) |

`docker-compose.yml` is a local stand-in with a 2 CPU / 4 GB cap. Its image tag is `tensorforge:dev`; retag or edit it to match the tag you built.

## Local development

Python 3.12. The service runs on CPU. Training the encoder wants an NVIDIA GPU (CUDA 12.x) or the Kaggle notebook in `notebooks/`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
copy .env.example .env
# set API_KEY in .env, then download the dataset into data/ (see data/README.md)
python scripts/verify_assets.py
python -m pytest
uvicorn app.main:app --env-file .env --host 0.0.0.0 --port 8000
```

On Linux or macOS: create a Python 3.12 venv, `pip install -r requirements-torch-cu126.txt` (or the CPU wheel), `pip install -r requirements-train.txt -r requirements-dev.txt`, then `sh scripts/install_hooks.sh`.

Contract check against a running service (local or hosted). The key is read from the environment or `.env` and is never printed.

```powershell
python scripts/e2e_check.py --base-url http://127.0.0.1:8000
python scripts/e2e_check.py --base-url https://tensorforge-fade.southindia.cloudapp.azure.com --big
```

`--big` adds a 5,000-ticket job. On a 2 CPU / 4 GB container on this laptop that job finished in 72 seconds at about 560 MB, and `/health` was 200 within 3 seconds of start. The earlier single-ticket and 100-ticket timings are in [`ml/reports/wp7.md`](ml/reports/wp7.md).

## Training

The serving model is a weighted average of a classical TF-IDF model (`svc_word_char`) and an int8 ONNX export of `intfloat/multilingual-e5-small` with task heads. Fusion weights, temperatures, and thresholds were fit on out-of-fold predictions. The encoder weights in the image are a final fit on all 4,800 tickets for 4 epochs.

```powershell
python -m ml.eda
python -m ml.train_classical
python -m ml.export_classical_oof
python -m ml.train_encoder
python -m ml.train_encoder --final --epochs 4
python -m ml.fuse
python -m ml.export_onnx
python -m ml.freeze --encoder
```

`ml/train_encoder.py` is the GPU run. `notebooks/kaggle_encoder.ipynb` is the same run as launched on Kaggle, and `notebooks/kaggle_encoder_executed.ipynb` keeps that run's output. Every experiment row is in [`ml/reports/experiments.csv`](ml/reports/experiments.csv). The classical choice is in [`ml/reports/baseline_decision.md`](ml/reports/baseline_decision.md), the error analysis in [`ml/reports/error_analysis.md`](ml/reports/error_analysis.md).

When an email subject and body name different issues, the scorer uses the body. A safety issue on either side wins, and a vague body such as "please help" follows the subject. Tickets with no subject, or where both sides agree, keep the combined score the model was trained on.

## API key

- The key is read only from the `API_KEY` environment variable at startup. It is never hard-coded, baked into the image, or committed. Locally it lives in the gitignored `.env`.
- A pre-commit hook (`.githooks/`, `core.hooksPath`) runs `scripts/check_secrets.py` and blocks a commit that contains a `tf2_...` key, the exact value from your local `.env`, an `.env` file, or organizer dataset files. CI (`.github/workflows/ci.yml`) runs the same scan.
- If `API_KEY` is unset, prediction and job endpoints return `401`. They do not run open.

## Dataset

The organizer dataset is not in this repo. Download it from the
[official link](https://drive.google.com/drive/folders/16S4yfPFzjbTQUb8uD8g0GPFCyK1LPWYm?usp=sharing)
into `data/` (see [`data/README.md`](data/README.md)). `scripts/verify_assets.py` checks that the spec, JSON Schemas, and dataset load and map onto each other.

## License

Classical TF-IDF models were trained for this project. The encoder is `intfloat/multilingual-e5-small` (MIT), exported to int8 ONNX. The same note is in `artifacts/manifest.json`.

## Layout

```
app/         FastAPI service (auth, validation, inference, jobs)
ml/          EDA, training, fusion, ONNX export, evaluation reports
artifacts/   frozen model files and manifest.json (model_version)
ui/          demo, served at /demo/
tests/       contract, auth, jobs, and model tests
spec/        organizer OpenAPI YAML and JSON Schemas
data/        organizer dataset (gitignored; download separately)
scripts/     setup, secret scan, asset check, end-to-end contract check
docs/        architecture and implementation plan
```

## Environment files

| File | Purpose |
| --- | --- |
| `requirements.txt` | Pinned runtime dependencies for the Docker image (CPU only) |
| `requirements-torch-cu126.txt` | CUDA 12.6 PyTorch, local training only |
| `requirements-train.txt` | Training tools (pandas, transformers, lightgbm, onnx, jupyter) |
| `requirements-dev.txt` | pytest, httpx, ruff |
| `requirements-train.lock.txt` | Exact versions of the working local environment |
| `.env.example` | Template for local configuration; copy to `.env` |
