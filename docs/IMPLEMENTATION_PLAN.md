# TensorForge 2.0 Phase 2 - Implementation Plan

Companion to [`ARCHITECTURE.md`](ARCHITECTURE.md). Architecture says **what and why**; this says
**what to build, in what order, how we know it is done, and when to cut**.

Deadline: **Saturday 10 Oct 2026, 18:00** (no extensions). Solo build. Day 0 (4 Oct) is done.

---

## 1. Working agreements

- **Vertical slice first.** By the end of Day 1 a thin end-to-end service (auth, validation, `/predict`
  with a baseline model) must run, so contract risk is found early and the model can improve behind a
  stable API.
- **Contract before cleverness.** If time is short, cut model sophistication, never contract tests.
- **Every change keeps `main` green.** Small commits; CI (secret scan, lint, spec check) must pass; the
  pre-commit hook stays enabled.
- **Priorities.** P0 = required to submit and score; P1 = clearly improves score; P2 = polish.
  When behind schedule, drop P2, then P1, never P0.
- **Freeze point.** Model frozen at the end of Day 4. After that: only bug fixes, docs, video.
- **Secrets.** `API_KEY` only via env / `.env`. Tests read it from the environment. Never print it.

---

## 2. Status

| Item | Status |
| --- | --- |
| Repo, hooks, CI, environment (Python 3.12 + CUDA torch), asset verification | done (Day 0), pushed to GitHub |
| Architecture and implementation plan | done |
| WP1 EDA, folds, metrics | done (Day 1) |
| WP2 classical baseline v0 (`svc_word_char`, Stage A macro-F1 0.656) | done (Day 1) |
| WP3 `/health`, `/predict`, `/predict/batch` and contract tests | done (Day 1) |
| WP4 job API (`/batch/jobs`, poll, results, delete) | done (Day 2) |
| WP5 fusion, ONNX, gate G1 | done: encoder adopted; final fit exported to int8 and fused in the engine |
| Encoder Stage A + 5-fold CV | done (Day 2); not the serving model |
| WP9 Azure | image `sha256:45131d2759b9a58d5d40529f5c740f2ee7c5ceb7464eef1b0a80d793dc31a3db` is running; `https://tensorforge-fade.southindia.cloudapp.azure.com/health` returns `v1.0.0-38ecb9bd` |
| Manifest and Docker image | `v1.0.0-38ecb9bd`, encoder on; image `tensorforge:dev` (994 MB) |
| WP7 local (2 CPU / 4 GB, laptop cores) | done: `/predict` p95 48 ms, 100-batch 1.9 s, 5,000-job 128 s, ~560 MB, startup 4.6 s, offline start 3.3 s. See `ml/reports/wp7.md` |
| WP7 on the Azure VM, WP9 deploy, WP10 demo | hosted image is live; 5,000-job succeeded; 2,000-job poll p95 under 1 s; demo at `/demo/` |

---

## 3. Work packages

Each package lists deliverables and **acceptance criteria** (how we know it is finished).

### WP1 - EDA and evaluation harness (P0, Day 1 morning)

Build:
- `ml/data.py`: loaders (csv module / pandas with `keep_default_na=False`), split helpers, `folds.json`
  generator (5 folds over train+validation, stratified on category x language, fixed seed).
- `ml/eda.ipynb` (outputs stripped before commit): label structure, secondary and urgency rules,
  language/channel effects, text lengths, noise, injection examples, resolved-complaint cases,
  quoted-correspondence cases, near-duplicate check.
- `ml/evaluate.py`: one function that takes true labels + predicted probabilities and returns all
  metrics (accuracy, macro-F1, per-class, confusion matrix, secondary metrics, urgent P/R/F1, ECE,
  Brier, slices by language and channel) and appends a row to `ml/reports/experiments.csv`.

Done when: `folds.json` is committed (ids only, no text); evaluating the trivial majority-class
predictor produces the full metrics table; EDA findings are written down as 10-15 bullet points that
feed feature and rule decisions.

### WP2 - Classical baseline v0 (P0, Day 1)

Build:
- `app/labels.py` and `app/text.py` (shared with serving).
- `ml/train_classical.py`: TF-IDF word + char n-grams; heads for category, secondary (OvR on 5
  classes), urgent; Stage A (train to validation) then CV with OOF output; saves artifacts to
  `artifacts/` and a metrics report.

Done when: Stage A and CV numbers are logged for at least three configurations (for example
LogReg vs LinearSVC vs calibrated SVC; with and without char n-grams); best config chosen with
a written reason; artifacts load in a fresh process and reproduce the metrics.

### WP3 - API skeleton and contract layer (P0, Day 1)

Build (see ARCHITECTURE section 4): `config`, `errors`, `middleware` (request id, auth, content type,
size, 404/405 JSON, last-resort 500), `auth`, `validation`, `routes` for `/health` and `/predict`,
`/predict/batch`, serving the baseline engine via `app/inference.py` (classical branch only).

Done when:
- `/health`, `/predict`, `/predict/batch` return responses that validate against the JSON Schemas for
  200 and error cases.
- The check order is proven by tests (401 beats 415 beats 413 beats 400 beats 422).
- Both auth styles work; unset `API_KEY` gives 401; wrong key gives 401 with `WWW-Authenticate`.
- Unknown route 404 and wrong method 405 are JSON.

### WP4 - Job subsystem (P0, Day 2)

Build `app/jobs.py` and the three job routes plus `DELETE` (ARCHITECTURE section 6).

Done when (all automated tests):
- submit gives 202 with `Location` and `Retry-After`; invalid item gives 422 and creates no job.
- polling shows monotonic `processed`; results paged correctly; 409 before success; 404 unknown;
  `DELETE` gives 204 then 404; expiry gives 410 (test with a shortened retention setting).
- idempotency returns the same job; the 5th active job gets 429 with `Retry-After`.
- killing and restarting the process mid-job leaves the job `failed` with `interrupted`.
- `/health` and polling stay under 1 s while a 5,000-ticket job runs; results are identical across runs.

### WP5 - Model improvement loop (P0 classical, P1 encoder; Days 1-3)

1. Error analysis on the baseline organized by the labelling rules (safety precedence, late-food refund,
   missing-food refund, OTP/login vs post-login, lost item, onboarding/payout, vague requests,
   resolved safety complaints, injection text, quoted resolved correspondence).
2. Classical improvements: feature variants (normalization options, channel and subject handling),
   class weights, calibration, extra members (SGD, ComplementNB, LightGBM on sparse features).
3. **Encoder** (`ml/train_encoder.py`, device-agnostic): multi-task heads (category, secondary OvR,
   urgent), head+tail truncation, frozen embeddings, fp16, class weights, early stopping on fold
   validation. Laptop smoke test, then Kaggle T4 for 5-fold CV and the final fit.
4. `ml/fuse.py`: fit fusion weights, temperatures and thresholds on OOF predictions.
5. `ml/export_onnx.py`: one ONNX graph with three heads, int8 dynamic quantization, parity check against
   PyTorch outputs on 200 tickets, tokenizer saved as `tokenizer.json`.

Done when: every experiment is a row in `experiments.csv`; the **encoder gate** (ARCHITECTURE section 7)
is evaluated and the decision is recorded; leave-one-language-out and robustness-slice results exist;
confidence calibration (ECE, reliability diagram) is reported.

**Decision gate G1 (end of Day 3):** encoder adopted only if it beats the classical ensemble by the gate
and meets the latency/memory budget on a 2 CPU, 4 GB limited container. Otherwise ship the improved
classical model as v1 and stop encoder work (it can return as v1.1 only if time remains before Day 5).

### WP6 - Engine integration and artifacts (P0, Days 3-4)

Build: `Engine` loading the manifest, hash verification, both branches, fusion, post-processing and
rules (ARCHITECTURE 5.2), fallback to classical on encoder exception, `model_version`. `ml/freeze.py`
writes `artifacts/manifest.json`.

Done when: the engine reproduces the offline OOF-style predictions on validation within tolerance;
consistency rules hold on 100% of outputs for all 4,800 dataset tickets and a fuzz set; the same ticket
gives identical output via `/predict`, `/predict/batch` and a job; model version string is identical
across all endpoints.

### WP7 - Performance and robustness verification (P0, Day 4)

Measure on the 2 CPU / 4 GB limited container: `/predict` p50/p95, 100-batch time, 2,000 and 5,000
ticket jobs, memory high-water mark, `/health` latency during a job, start-up time to 200.
Fuzz (see test matrix). Fix hot spots (thread counts, tokenization, caching of vectorizer).

Done when: all spec latency targets are met with margin, no 5xx under fuzz, memory stays under about 2 GB.

### WP8 - Docker image (P0, Day 3-4)

Build: multi-stage `Dockerfile`, `.dockerignore`, build-time self-test, `docker-compose.yml` for local
convenience (optional). Healthcheck.

Done when: `docker build` then `docker run -p 8000:8000 -e API_KEY=<key> <image>` works; also with
`--network none`; also under `--cpus 2 --memory 4g`; `docker history` and layers contain no key; image
size is recorded; `/health` is 200 well under 120 s.

### WP9 - Azure hosting (P0, start Day 1, finish Day 4)

Day 1 (in parallel, 30-45 min): create VM (check quota), static IP, DNS label, NSG rules, install Docker
and Caddy, serve a placeholder `/health` over HTTPS so DNS/quota/cert problems appear early.
Day 4: push the image to a registry, pull by digest, run with `--restart unless-stopped`, env file for
the key, Caddy limits (body above 25 MB, timeouts), uptime monitor, 2 GB swap, auto-shutdown off.

Done when: `https://<label>.<region>.cloudapp.azure.com/health` is 200 with a valid certificate; the
full hosted test suite (including a 5,000-ticket job) passes against it using the real key; the digest
is recorded; the VM survives a reboot test.

### WP10 - Demo UI (P1, Day 4; P2 polish Day 5)

Plain HTML/JS/CSS under `ui/`, served at `/demo/`. Key-gated (visitor pastes the key; kept only in
session storage). Features: live single-ticket predictor with sample tickets in each language and
channel, confidence bar, team, secondary and urgency; CSV upload that calls the batch/job flow with a
results table and download; metrics page with pre-rendered confusion matrix, per-class and per-language
scores and the calibration plot (images generated by `ml/evaluate.py`).

Done when: all of the above works on the hosted URL; no key in page source; works in a fresh browser.

### WP11 - Documentation, report, video (P0, Days 4-5)

- `README.md`: exact `docker run` command, local dev, tests, training reproduction, `needs_human_review`
  threshold, hosted URL and image digest, model version, license notes.
- `docs/report.md` (max 5 pages): problem and data, EDA findings, experiment table, final model,
  evaluation scores, error analysis, API design, limitations, honest findings.
- `docs/video_script.md` and the recording (15 minutes, YouTube unlisted): solution, architecture,
  training and validation story (this is where "not just an LLM" is shown), live demo, evaluation.

Done when: report is within 5 pages as PDF; video link works while logged out; README commands tested
on a clean clone.

### WP12 - Freeze, final checks, submit (P0, Day 6)

Run the pre-submission checklist (section 7), tag the release, submit with a buffer of at least six hours.

---

## 4. Schedule

| Day | Date | Focus | Exit criteria (end of day) |
| --- | --- | --- | --- |
| 0 | Sun 4 Oct | Scaffolding, environment, plans | done |
| 1 | Mon 5 Oct | WP1, WP2, WP3, start WP9 | vertical slice runs locally with baseline model; contract tests started; Azure placeholder reachable over HTTPS |
| 2 | Tue 6 Oct | WP4, WP5 (error analysis, classical tuning, Kaggle CV **launched first thing**) | jobs complete with tests; classical v1 candidate; encoder CV running or finished |
| 3 | Wed 7 Oct | WP5 (fusion, ONNX), WP6, WP8 start | **gate G1 decided**; engine integrated; Dockerfile builds |
| 4 | Thu 8 Oct | WP6 finish, WP7, WP8, WP9 deploy, WP10; **model freeze** | hosted service passes full suite and a 5,000-ticket job; manifest and digest recorded |
| 5 | Fri 9 Oct | WP10 polish, WP11 (report, video) | README, report PDF, video uploaded |
| 6 | Sat 10 Oct | WP12 | final checklist green; **submit by 12:00**, hard deadline 18:00 |

Buffers: Day 5 afternoon is spare; if Day 4 slips, drop WP10 polish and encoder v1.1, never WP7/WP11.

**Cut list (in order):** UI polish, extra ensemble members, encoder (fall back to classical), CSV upload
in the UI, metrics page in the UI, optional `meta` fields. **Never cut:** contract tests, jobs,
Docker offline run, hosted deployment, training story, report, video.

---

## 5. Test matrix (`tests/`)

| Area | Tests |
| --- | --- |
| Contract (jsonschema) | every 200 response validates against its schema; error bodies validate against `error_response`; team/secondary/spam rules hold for every output (dataset tickets + fuzz) |
| Auth | no key, wrong key, both header styles, empty `API_KEY`, key with whitespace, timing-safe compare used, `/health` public, auth beats 415/413/400/422, 404/405 JSON |
| Check order | combinations such as bad content type + bad JSON, huge body + wrong key, wrong key + valid body |
| Validation | empty/whitespace/emoji/mixed-script text, 10,000 vs 10,001 chars, subject 500 vs 501, bad channel, wrong types, nulls, missing fields, extra fields, top-level array, deep nesting, non-UTF-8 bytes, batch size 0/1/100/101, duplicate and missing `ticket_id`, many errors with correct `index` |
| Differential | our validators vs `jsonschema` on thousands of generated/mutated payloads |
| Batch | order preserved, ids echoed, atomic failure, independence (shuffle a batch, outputs identical per ticket), `meta.count` |
| Jobs | lifecycle, paging, idempotency, 429, 409, 410, 404, `DELETE`, restart-interrupted, determinism, responsiveness during a job, retention sweep |
| Concurrency | 20 parallel `POST /batch/jobs` with the same `Idempotency-Key` give one job id and no 500; parallel submits beyond capacity give exactly 1 running + 3 queued and 429 for the rest; `/health` p95 under 1 s while a 5,000-ticket job runs under a mixed `/predict` load (measured on the 2 CPU / 4 GB limited container) |
| Robustness | random bytes, huge unicode, control characters, injection phrases, 10k-char text, concurrency (50 parallel `/predict`), repeated 5,000 jobs |
| Model | gold-label regression thresholds on validation (fail the build if macro-F1 drops), parity of ONNX vs PyTorch, calibration sanity |
| Hosted (`-m hosted`) | same suite against `TF_BASE_URL` with the real key from the environment, including a 5,000-ticket job |
| Image | start with `--network none`, `/health` time, resource-limited run, no key in layers |

---

## 6. Spec ambiguities: decided (no need to ask the organizers)

Each of these is safe whatever the organizers would answer, so we implement the default and document it
in the README. Only ask the organizers if something *blocks* a decision.

| Topic | Decision |
| --- | --- |
| Extra request fields such as `language` | Accepted and ignored. If present it is `Optional[str] = None` style: never required, never validated beyond "ignore", never a model feature (the spec says the model sees only channel, subject, text). Must not crash when omitted. |
| `HEAD /health` | Supported together with `GET /health`. |
| Wrong method on a protected path without a key | 401 first (auth before anything else). 405 JSON only when the key is valid. |
| `Idempotency-Key` reused with a different payload | Return the original job (202, same `job_id`); never create a duplicate. |
| Reference hardware ("to be confirmed" in the spec) | Assume 2 vCPU / 4 GB (Standard_B2s), no GPU. |
| `confidence` meaning | Calibrated probability of the primary `category` only (schema text). `needs_human_review` is the optional bonus field: set when confidence is below a documented threshold. |

Rule of thumb: ask the organizers only when a wrong guess would lose points *and* we cannot pick a
default that is safe under every plausible answer.

---

## 7. Final checklist (Day 6)

Contract and service
- [ ] `scripts/verify_assets.py` passes; JSON Schema contract tests pass locally and on the hosted URL
- [ ] All protected endpoints require the key; `/health` public; both auth styles work
- [ ] No 5xx under the fuzz suite; unknown routes/methods return JSON
- [ ] 5,000-ticket job succeeds on the hosted URL within 30 minutes; polling stays responsive
- [ ] `model_version` identical in `/health`, `/predict`, `/predict/batch`, job status and results

Container and hosting
- [ ] `docker run -p 8000:8000 -e API_KEY=<key> <image>` works on a clean machine, offline (`--network none`)
- [ ] Hosted URL has a valid certificate, survives reboot, runs the recorded image digest
- [ ] Uptime monitor is green; Azure credits cover the evaluation window; VM auto-shutdown is off

Repository and evidence
- [ ] Public repo: code, training scripts, evaluation scripts, experiment log, tests, Dockerfile, README
- [ ] No key, `.env`, dataset or booklet in git history (`check_secrets.py --all` clean; history reviewed)
- [ ] Training and validation process clearly documented (not "an LLM call")
- [ ] Report PDF (max 5 pages) with design decisions, evaluation scores, findings
- [ ] Video (15 minutes, YouTube unlisted) opens while logged out
- [ ] Demo link works in a fresh browser (key-gated, key pasted by the visitor)
- [ ] Release tag (for example `v1.0.0`) matches `model_version` in the manifest

---

## 8. Risk register and fallbacks

| Risk | Likelihood | Impact | Mitigation / fallback |
| --- | --- | --- | --- |
| Contract detail missed | medium | high | contract tests from Day 1, differential validation test, ask organizers early |
| Encoder too slow/large or not better | medium | medium | gate G1; ship improved classical model |
| Kaggle quota, queue or session loss | medium | medium | script is resumable, checkpoints per fold; laptop fallback with smaller model |
| Encoder CV takes longer than hoped | medium | medium | realistic estimate is 15-25 min per run on a T4, so 5 folds plus the final fit is about 1.5-2.5 h (session limit 12 h, weekly quota 30 h). Write `train_encoder.py` and smoke-test it on the laptop by the end of Day 1 so CV can start first thing on Day 2; run folds as separate checkpointed steps |
| Hidden set differs in language mix | medium | medium | char n-grams, leave-one-language-out checks, calibrated confidence |
| 4 GB memory pressure with job running | low | high | one model copy, int8, measured in WP7, swap file |
| Azure quota/credits/DNS delay | medium | high | start Day 1, placeholder HTTPS early, check credit end date |
| Certificate or proxy returns non-JSON errors | low | medium | Caddy limits above the app's, test oversized bodies through the proxy |
| Key leaked | low | high | hooks + CI scan + history check; report to organizers, do not rotate alone |
| Solo bandwidth | high | high | strict cut list, UI last, report and video outline early |

---

## 9. Where things stand (morning of Day 2, Tue 6 Oct)

Re-checked this morning: the test suite passes. Day 1 WP1–WP3 are in the tree (folds, EDA
findings, three classical configs, serving artifact `artifacts/classical.joblib`, contract tests).

Day 2 exit is met in the working tree:

- Job API and `tests/test_jobs.py`: submit 202, invalid item creates no job, monotonic `processed`,
  paging, 409 / 404 / 410, idempotency, 429, restart `interrupted`, health and poll stay under 1 s
  during a 5,000-ticket job.
- Classical search kept `svc_word_char` (v0.1.0), including the LightGBM member (Stage A macro-F1 0.475). See `ml/reports/classical_v1.md`.
- Encoder Stage A macro-F1 0.786 and the 5-fold numbers are in `ml/reports/experiments.csv`.
  The Kaggle output (summary, stage A, and fold probabilities, no weights) is in
  `ml/runs/encoder_e5_small`. Pass the token as `KAGGLE_API_TOKEN`; `KAGGLE_KEY` alone gets HTTP 401.

Checked after Azure sign-in: the South India VM `tensorforge` was already up (`Standard_B2als_v2`,
2 vCPU / 4 GB, static IP, DNS `tensorforge-fade.southindia.cloudapp.azure.com`). Docker and Caddy
are installed. `GET /health` and `HEAD /health` return 200
`{"status":"ok","model_version":"placeholder","model_loaded":false}` with a valid certificate.
`TF_BASE_URL` is set in `.env`. SSH is limited to the current public IP; ports 80 and 443 are open.

Day 3 exit is met in the working tree:

- Gate G1 passed (`ml/reports/g1.md`). Under a 2 CPU / 4 GB cap, validation p95
  was 36 ms and a 10k-character ticket p95 was 73 ms, at 499 MB.
- The serving encoder is a final fit on all 4,800 tickets for 4 epochs (the
  stage-A best epoch), not the smoke checkpoint. `encoder_enabled` is true.
  Model version is `v1.0.0-38ecb9bd`.
- The int8 file is 118 MB, over GitHub's 100 MB limit, so it is not committed.
  The image copies it from the local `artifacts/` directory.
- Fused labels match the fp32 graph on 200 validation tickets.

Day 4 local performance is in `ml/reports/wp7.md`. The same image under
`--cpus 2 --memory 4g --memory-swap 4g` met the spec targets on this laptop
(not the VM's cores): `/predict` p95 48 ms, 100-batch 1.9 s, 2,000-job 53 s,
5,000-job 128 s, health p95 about 27 ms during the job, resident memory 563 MB,
startup 4.6 s. `--network none` reached `/health` 200 in 3.3 s. Image size is
994 MB. `docker history` has no API key.

The Azure host is serving the frozen image. `https://tensorforge-fade.southindia.cloudapp.azure.com/health` returns `v1.0.0-38ecb9bd`. The container is limited to 2 CPUs and 3 GB because the VM has 3.8 GB of RAM, with a 2 GB swap file. A 5,000-ticket job on the VM reached `succeeded` at 5,000/5,000, and resident memory stayed about 557 MB.

On 9 October 2026 the running image was replaced with `sha256:45131d2759b9a58d5d40529f5c740f2ee7c5ceb7464eef1b0a80d793dc31a3db` (`tensorforge:v1.0.1`, built from commit `52e612b`). The previous image `sha256:e18b0c716e0240c4dc86f9a04703380e9a82cd2a604856e4d30d3e21da88a11c` is still on the VM as the rollback image, and the job volume `tf_data` was kept. An earlier image id `sha256:5d64978226bc891c3d7b09abf12ec19eb25d0c16aa8e523b6094d284b0639920` is not the deployed release. SSH is limited to the operator's current address.

Auto-shutdown is off. A reboot on 6 Oct 2026 brought the container back healthy in under a minute, and the public `/health` still returned `v1.0.0-38ecb9bd`. A later 2,000-ticket job on the VM had `/health` p95 844 ms and poll p95 761 ms, with no failed samples (`ml/reports/wp7_azure_poll.md`). The demo is at `/demo/`.
