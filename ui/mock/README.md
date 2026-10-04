# TensorForge Dev-Only Mock Server

This directory provides a lightweight Express.js mock server implementing all 7 endpoints of the TensorForge Phase 2 OpenAPI specification.

## Usage

```bash
# Start the mock server independently
npm run mock

# Or run Vite + Mock server concurrently
npm run dev
```

## Endpoints

- `GET /health` — Public health status
- `POST /predict` — Single ticket classification
- `POST /predict/batch` — Batch classification (1–100 tickets)
- `POST /batch/jobs` — Async job submission (up to 5,000 tickets)
- `GET /batch/jobs/:job_id` — Job status & progress polling
- `GET /batch/jobs/:job_id/results` — Paginated job results
- `DELETE /batch/jobs/:job_id` — Job cancellation

## Authentication

All endpoints except `/health` require an API key passed via:
- Header `X-API-Key: test-key-dev` (default)
- Or `Authorization: Bearer test-key-dev`

To configure a custom key:
```bash
MOCK_API_KEY="my-secret-key" npm run mock
```
