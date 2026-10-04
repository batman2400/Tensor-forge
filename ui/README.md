# TensorForge 2.0 — Dispatch Desk UI

The **Dispatch Desk** is the web dashboard for TensorForge 2.0, providing an interactive, typographic, paper-and-ink interface for real-time customer support ticket classification and dispatch routing on the RideEat platform.

---

## Features

- **Ink-on-Paper Typographic Design**: Pure CSS design tokens adhering to the design mock (`--paper`, `--ink`, `--stamp`, hairline rules, stamp-in animations).
- **Self-Hosted Open Fonts**: Inter, Newsreader, and JetBrains Mono bundled directly via `@fontsource` packages. **Zero external CDN dependencies**.
- **Three Core Operational Zones**:
  1. **Single Ticket (`POST /predict`)**: Instant classification, channel selection (`chat`, `email`, `call_transcript`), optional subject, animated stamp, route arrow, confidence track, and view JSON / cURL modal.
  2. **Batch Classification (`POST /predict/batch`)**: Synchronous processing of 1–100 tickets, real-time ticket counting, client-side sorting by any column, "Urgent only" filtering, and CSV export.
  3. **Async Batch Jobs (`POST /batch/jobs`)**: Handles up to 5,000 tickets via drag-and-drop file upload (`.csv`, `.jsonl`, `.json`), automatic polling lifecycle (`queued` → `running` → `succeeded`), progress indicator, cancellation support, localStorage session recovery, and CSV/JSON downloads.
- **Session Security**:
  - Full-screen Unlock overlay verifying API credentials via `POST /predict` before granting access.
  - API keys stored strictly in tab `sessionStorage` (never persisted to disk).
  - Immediate session lock on any HTTP 401 error or explicit user lock action.
  - Safe cURL / JSON panel with redacted key placeholder (`X-API-Key: $API_KEY`).
- **Autonomous Dev Mock Server**: Complete Express.js mock server implementing all 7 endpoints with realistic classification heuristics and progress simulation.

---

## Directory Structure

```
ui/
├── mock/                        # Development-only mock server
│   ├── data.ts                  # Heuristic classification rules and sample logic
│   ├── handlers.ts              # Handlers for all 7 endpoints with auth
│   ├── server.ts                # Express server on port 3001
│   └── README.md                # Mock server documentation
├── public/
│   └── favicon.svg              # TF monogram icon
├── src/
│   ├── api/
│   │   ├── client.ts            # Fetch wrapper with auth injection & 401 interceptor
│   │   ├── types.ts             # OpenAPI contract TypeScript definitions
│   │   ├── health.ts            # GET /health
│   │   ├── predict.ts           # POST /predict
│   │   ├── batch.ts             # POST /predict/batch
│   │   └── jobs.ts              # /batch/jobs endpoints
│   ├── components/
│   │   ├── AsyncJob/            # Zone 03: Async jobs, dropzone, progress
│   │   ├── BatchPredict/        # Zone 02: Batch textarea, sortable table
│   │   ├── SinglePredict/       # Zone 01: Single classify, result card
│   │   ├── ErrorBanner.tsx      # Standardized alert & validation error display
│   │   ├── Header.tsx           # Brand, health indicator, theme toggle, lock button
│   │   ├── LoadingSpinner.tsx   # Minimal ink loading indicator
│   │   └── UnlockScreen.tsx     # API key gate & probe
│   ├── hooks/
│   │   ├── useApiKey.ts         # Session storage API key management
│   │   ├── useHealth.ts         # Health polling hook
│   │   ├── useJobPoller.ts      # Polling lifecycle & localStorage recovery
│   │   └── useTheme.ts          # Dark/light theme switcher
│   ├── lib/
│   │   ├── categories.ts        # 11 categories & team routing mapping
│   │   ├── download.ts          # CSV and JSON client-side export
│   │   ├── format.ts            # Display formatting helpers
│   │   └── samples.ts           # Hand-written sample tickets
│   ├── __tests__/               # Vitest test suite
│   ├── App.tsx                  # Root application component
│   ├── index.css                # Design system tokens, typography & reset
│   └── main.tsx                 # React entry point
├── index.html                   # HTML entry point
├── package.json                 # Dependencies & build scripts
├── tsconfig.json                # TypeScript compiler config
└── vite.config.ts               # Vite proxy & build configuration
```

---

## Development

### 1. Install Dependencies
```bash
cd ui
npm install
```

### 2. Run in Mock Mode (Default)
Starts both the dev mock server (port 3001) and Vite dev server (port 5173) concurrently:
```bash
npm run dev
```
Open [http://localhost:5173/demo/](http://localhost:5173/demo/) in your browser.
Default dev API key: `test-key-dev`

### 3. Run with Real FastAPI Backend
When the FastAPI backend is running (e.g. on `http://localhost:8000`):
```bash
VITE_API_BASE_URL=http://localhost:8000 npm run dev:real
```

---

## Testing & Quality

Run the test suite with Vitest:
```bash
npm test
```

Typecheck without emitting files:
```bash
npx tsc --noEmit
```

---

## Production Build

```bash
npm run build
```
Build output is generated in `ui/dist/` (static HTML, CSS, JS, and font assets). Total bundle size is under 100 KB gzipped.

---

## Backend Integration Instructions

### 1. FastAPI Static Mount
In `app/main.py`, mount the static bundle under the `/demo` route:

```python
import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles

# Register after all API routes (/predict, /health, /batch/jobs, etc.)
ui_dist_path = Path(os.environ.get("UI_DIST_PATH", "ui-dist"))
if ui_dist_path.is_dir():
    app.mount("/demo", StaticFiles(directory=str(ui_dist_path), html=True), name="demo-ui")
```

### 2. Multi-Stage Dockerfile
In the root `Dockerfile`, build the UI in a Node builder stage and copy only the compiled static files into the Python runtime container:

```dockerfile
# --- Stage 1: Build UI ---
FROM node:22-alpine AS ui-build
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ .
RUN npm run build

# --- Stage 2: Python Runtime ---
FROM python:3.12-slim
WORKDIR /app
# ... install python requirements ...
COPY . /app
COPY --from=ui-build /ui/dist /app/ui-dist

ENV UI_DIST_PATH=/app/ui-dist
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
Node.js and npm are **never** included in the production image.
