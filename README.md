# TensorForge 2.0 - Phase 2 MVP: Support Ticket Classification and Routing

Team **Fade**. A self-trained (no LLM wrapper) classifier for the fictional RideEat ride-hailing and
food-delivery platform. For each support ticket it predicts:

- `category` (11 classes) and the routed `team` (fixed mapping from the organizers' spec)
- `secondary_category` (second issue, or `null`)
- `is_urgent`
- a calibrated `confidence` for the primary category

The service follows the organizers' OpenAPI contract in [`spec/`](spec/): `POST /predict`,
`POST /predict/batch` (1-100 tickets), async `POST /batch/jobs` (up to 5,000), job polling/results,
and a public `GET /health`. All endpoints except `/health` require the API key.

> **Status: Day 0 (scaffolding).** Repository, environment, git hooks and asset verification are in
> place. EDA, model training and the API are built from Day 1. This README grows with the project
> (training story, evaluation scores, architecture and the exact `docker run` command).

## Quick start (development)

Requirements: Windows/Linux/macOS, Python 3.12 (installed for you via [uv](https://docs.astral.sh/uv/)),
an NVIDIA GPU for training (CUDA 12.x driver). The service itself runs on CPU only.

```powershell
# 1. Create the environment (venv lives outside OneDrive), install CUDA PyTorch + deps, enable git hooks
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

# 2. Put your key in .env (copied from .env.example; gitignored)
#    API_KEY=<your key>

# 3. Download the dataset into data/ (see data/README.md), then verify everything
python scripts/verify_assets.py
```

Linux/macOS: create a Python 3.12 venv, `pip install -r requirements-torch-cu126.txt` (or the CPU
wheel), `pip install -r requirements-train.txt -r requirements-dev.txt`, then `sh scripts/install_hooks.sh`.

## Environment files

| File | Purpose |
| --- | --- |
| `requirements.txt` | Pinned runtime dependencies for the Docker image (CPU only) |
| `requirements-torch-cu126.txt` | CUDA 12.6 PyTorch, local training only |
| `requirements-train.txt` | Training/research tools (pandas, transformers, lightgbm, onnx, jupyter) |
| `requirements-dev.txt` | pytest, httpx, ruff |
| `requirements-train.lock.txt` | Exact versions of the working local environment |
| `.env.example` | Template for local configuration; copy to `.env` |

## Security: the API key

- The key is read **only** from the `API_KEY` environment variable at startup. It is never
  hard-coded, baked into the image or committed. Locally it lives in the gitignored `.env`.
- A pre-commit hook (`.githooks/`, enabled by `core.hooksPath`) runs `scripts/check_secrets.py` to
  block any commit containing a `tf2_...` key, the exact value from your local `.env`, `.env` files, or
  organizer dataset files. The same scan runs in CI (`.github/workflows/ci.yml`).
- Without `API_KEY` set, prediction endpoints answer `401` instead of running open.

## Dataset

The organizer dataset is not redistributed here. Download it from the
[official link](https://drive.google.com/drive/folders/16S4yfPFzjbTQUb8uD8g0GPFCyK1LPWYm?usp=sharing)
into `data/` (details in [`data/README.md`](data/README.md)). `scripts/verify_assets.py` checks that
the spec, JSON Schemas and dataset load and map onto each other.

## Repository layout

```
app/        FastAPI service (auth, endpoints, jobs, inference)      [Day 1+]
ml/         EDA, training, evaluation, ONNX export, reports          [Day 1+]
artifacts/  frozen model files + manifest.json (model_version)
ui/         demo dashboard served by the same container              [Day 5]
tests/      contract (JSON Schema), auth, robustness, job, load tests
spec/       organizer OpenAPI YAML + JSON Schemas
data/       organizer dataset (gitignored; download separately)
scripts/    setup_env.ps1, verify_assets.py, check_secrets.py, hooks installer
docs/       organizer guidelines, report and video script
```

## Deployment target

Single Docker container (`docker run -p 8000:8000 -e API_KEY=<key> <image>`), no internet needed at
runtime, behind Caddy (automatic HTTPS) on an Azure Ubuntu VM (2 vCPU / 4 GB) using the default
`<label>.<region>.cloudapp.azure.com` DNS name. Details are added with the Docker/deploy work.
