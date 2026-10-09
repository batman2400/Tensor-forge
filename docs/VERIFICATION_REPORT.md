# Verification report — 9 October 2026

**Update, same day, after the release checks.** Image `sha256:45131d2759b9a58d5d40529f5c740f2ee7c5ceb7464eef1b0a80d793dc31a3db` was built from commit `52e612b` and is the image running on the VM. Local container checks: startup 4.3 s, offline startup 3.3 s, 256/256 contract checks, 5,000-ticket job 80 s, peak memory 572 MiB, restart marks the job `interrupted`. From the VM through public HTTPS: single-prediction p95 0.035 s, 5,000-ticket submit 0.031 s and completion 119 s, poll p95 0.012 s. A 30 MB unauthenticated upload through Caddy returned JSON 401 in 0.15 s, and an authenticated 6 MB batch returned JSON 413. CI run 37921671034 passed. Tag `v1.0.0` was not moved. Release `v1.0.1` records this digest. Hosting-credit balance was not visible to the Azure login. The video is still not recorded, so this is not a 100% submission sign-off.

**Morning verdict: local functionality passes the exercised checks; the project is not ready for a 100% requirements-compliance sign-off.** The hosted upload and latency failures below were measured from the laptop before the VM retest. Passing tests do not establish perfect predictions or cover every possible input.

## Scope and identity

- Reviewed the local OpenAPI v2.1.0 contract, JSON Schemas, README, architecture, implementation plan, application, tests, model loading/fusion, job service, demo and evaluation evidence.
- Starting commit: `cc6da4fba7b0bf3959d2cc64fc1827483a3f0f8b`; model: `v1.0.0-38ecb9bd`.
- Local Python: 3.12.15 from the existing project environment. The default system Python is 3.14 and was not used to validate the service.
- Hosted target: `https://tensorforge-fade.southindia.cloudapp.azure.com`. The user explicitly approved authenticated testing. No key is included in this report.
- The original competition booklet is not present in the tracked repository. This audit cannot certify additional requirements that are absent from the available specification and plan.
- Local fixes below are uncommitted and have not been deployed. Existing user report PDFs and the video script were not modified.

## Verified results

- **112/112 tests passed**, no skips, in 244.50 seconds. Includes API authentication and error precedence, schema/validation boundaries, differential validation, job lifecycle, parallel idempotency, capacity, cancellation, expiry, restart recovery, prediction consistency, all 4,800 dataset tickets, fuzz inputs, encoder fallback, token handling, manifest and saved-prediction checks. Some API/concurrency tests use deterministic fake engines; real-model coverage is provided separately by serving tests and HTTP runs.
- **111 passed, one slow test deselected** after the edits, in 8.57 seconds.
- **Clean exported checkout: 107 passed, four dataset-dependent tests skipped, one slow test deselected**, in 9.80 seconds. Used committed file bytes plus the proposed edits; locally verified encoder bytes hydrated the LFS pointer. No private dataset or `.env` was copied. This tests clean-checkout code/model loading with the existing environment, not a fresh dependency installation or remote LFS download.
- **258/258 local HTTP checks passed** using the real model. Includes both authentication styles, multilingual input, malformed/oversized input, 100-ticket batch, order independence, idempotency, pagination, queue overflow, job cleanup and identical predictions across API paths.
- Local 2,000-ticket job: **39 seconds**. Local 5,000-ticket job: **112 seconds**, polling p95 approximately **0.01 seconds**. Local capacity response sequence: `202, 202, 202, 202, 429, 429, 429`.
- Separate short-ticket sample: local single prediction p95 **0.0241 seconds**, 30 requests; maximum of five 100-ticket batches **0.668 seconds**; **50 parallel local predictions** succeeded and were identical. These are laptop process measurements, not a 2-vCPU/4-GB container benchmark.
- Browser: sample prediction rendered category/team/confidence/urgency/version; a 101-row CSV with quoted multiline text completed through the async job flow; the downloaded CSV contained all 101 IDs in order; the metrics page populated its data and corrected evaluation note.
- Asset verification: **zero failures, zero warnings**. Schemas, OpenAPI examples, all 4,800 dataset requests and label mappings agree; CSV/JSONL contents match.
- Current committed classical model, fusion settings, tokenizer, encoder metadata and encoder LFS SHA-256 all match the committed manifest. Real `Engine.load()` succeeds.
- Recomputed fusion Stage A macro-F1 from saved branch probability files and frozen settings: **0.8021689933368322**. This reproduces the recorded score, not the original model training.
- Lint and whitespace checks passed. Current tracked-file secret scan passed. A scan of **166 reachable historical blobs** found no matches for the project's key patterns/current local secret or prohibited paths. This is not a general-purpose scan for every possible credential type.
- Installed dependency compatibility check: **161 packages compatible**.

## Hosted failures and limits

The complete hosted HTTP run finished with **247/253 checks passing**. Counts differ from the local run because some checks are conditional on timing and queue responses; these counts are not completion percentages.

1. Unauthenticated 30-MB uploads to each of `/predict`, `/predict/batch` and `/batch/jobs` timed out while writing instead of producing an observable 401 response: **three failures**.
2. An approximately 6-MB batch upload did not yield an observable JSON 413 response: **one failure**.
3. The 2,000-ticket submission exceeded the five-second acceptance target: **one failure**.
4. The 5,000-ticket submission returned 202 but exceeded five seconds: **one failure**.

The 2,000-ticket job completed in **37 seconds** and the 5,000-ticket job in **124 seconds**. Both are within job-completion budgets. Health p95 during the smaller job was approximately **0.41 seconds**. The harness's smaller-job `poll p95` value of **1.01 seconds** combines prediction and poll samples, so it must not be presented as a pure polling metric. The larger-job polling p95 was **21.80 seconds**, exceeding the one-second target.

A separate sample of 30 hosted short-ticket predictions measured p95 **1.891 seconds**, also exceeding the one-second target. Five 100-ticket batches had maximum elapsed time **2.733 seconds**; this small sample is not a robust batch-p95 estimate.

All hosted timings include the client network, TLS and reverse proxy. The failures are real observations from this connection, but their cause has not been isolated to application code. The local equivalents pass. Repeat from a stable client near the server and inspect reverse-proxy/network/server timings before changing application behavior or declaring compliance. Do not silently increase timeouts and relabel the original failures as passes.

The hosted run began before the two harness assertion fixes below; its observed responses also satisfy the corrected no-ID and overflow predicates. Its six recorded failures remain failures. The harness reports some latency statistics without asserting their targets, so the 247/253 headline does not include every performance nonconformance.

## Requirement coverage and outstanding evidence

- **API contract and model rules: passed exercised tests.** All seven required operations are implemented. Authorization precedes parsing; unset keys fail closed; mapping and secondary/spam invariants hold. Expiry/restart behavior is tested locally with shortened retention and simulated restarts, not by waiting six hours or rebooting the production VM.
- **Training and evaluation: implementation/evidence present, limitations remain.** Fusion tuning uses OOF predictions and labels over all 4,800 records, including validation labels. The 0.8022 score is development validation, not an untouched holdout. The final model is trained on all 4,800 records; the observed 800-row API macro-F1 of approximately 0.994 is in-sample and must not be marketed as validation accuracy. No full retraining, independent hidden-set evaluation, new ONNX/PyTorch parity run, or leave-one-language-out retraining was performed. The plan requests leave-one-language-out evaluation; the calibration report explicitly states it is absent.
- **Container/offline/resource targets: not freshly verified.** Docker CLI is installed, but the Linux engine pipe was unavailable. Starting Docker Desktop did not make the engine usable, and its WSL distribution remained stopped. Therefore no new build, offline startup, 120-second container startup, 2-vCPU/4-GB cap, memory high-water measurement, or layer scan is claimed. Historical benchmark files are supporting records only.
- **Hosted availability: partially verified.** HTTPS health and authenticated predictions/jobs succeeded; upload and latency failures are detailed above. Matching model-version strings do not prove matching image digests. The deployed image digest, reverse-proxy configuration, reboot survival, monitoring, credits and auto-shutdown settings were not inspected. Local and hosted examples have differing confidence values despite the same model version; platform-dependent int8 numerics or deployment differences require direct image/runtime comparison before explaining the difference.
- **Demo: verified local browser workflows.** No production-browser/mobile-browser sweep was performed. The test suite additionally confirms public demo assets and absence of a hardcoded key in those assets.
- **Report: present but needs reconciliation.** `TensorForge_Phase2_Final_Report.pdf` has five pages and was visually inspected. Its page 5 refers to older commit `f164ba4...` and describes artifact hash failures that are fixed in current commit `cc6da4f...`; it also says authenticated testing was unavailable, now superseded by this audit. A second report, `TensorForge_Phase2_Final_Report (1).pdf`, appeared during the audit and was not reviewed. Choose and update the intended final report before submission.
- **Video/submission: not verified.** The script exists, but no final YouTube URL was found in the README/docs. Recording length, logged-out access and actual submission cannot be confirmed.
- **Release identity: needs reconciliation.** Tag `v1.0.0` points to `f164ba49b01626e8926335e46639fd2655fc076e`, before the artifact-byte fix at current HEAD. The plan and README retain different image digests. Do not submit the older tag as evidence of the current verified tree without resolving this difference.
- **CI: improved locally, remote execution pending.** Added a Linux test job with LFS checkout, runtime/dev dependencies, non-slow tests and JUnit upload. Private-data-dependent saved-prediction tests now skip explicitly when the dataset is absent. The matching clean-checkout test passed locally; the new GitHub job has not been pushed/run.

## Fixes made during verification

1. Corrected a precedence bug that could make the no-ticket-ID HTTP check pass for an unsuccessful response.
2. Removed a capacity-check expression that injected a synthetic 429 and could conceal missing overflow behavior.
3. Added actual test execution to CI; retained existing lint/schema/secret checks.
4. Corrected the README, metrics page, fusion report and report generator to describe tuning-exposed validation accurately.
5. Corrected the README's contradictory claim that model weights were absent from Git; the encoder is tracked through LFS.
6. Replaced the demo's Singaporean-English “Singlish” example with romanized Sinhala and its English-only “Mixed” example with mixed-script text.

## Evidence and reproduction

Generated logs under `runtime/` are local, ignored verification evidence:

- `verification-tests.xml`: full 112-test run.
- `verification-fast-tests.xml`: post-edit non-slow run.
- `verification-clean-tests.xml`: clean checkout without private data.
- `verification-local.log`: 258/258 real HTTP checks.
- `verification-hosted.log`: 247/253 hosted checks and failure details.
- `verification-latency.json`: independent short-ticket performance sample.
- `verification-history.json`: historical scan count/findings.
- `verification-demo.jpg`: browser metrics-page evidence.

Use the project's Python 3.12 environment to rerun `python -m pytest`, `python -m ruff check .`, `python scripts/verify_assets.py`, and `python scripts/check_secrets.py --all`. With a running service, use `python scripts/e2e_check.py --base-url <url> --big`; the hosted command uses the configured private key. Preserve failed-run evidence when repeating checks.

**Release blockers:** diagnose/retest hosted upload and latency failures; restore Docker and prove the final image offline under the resource cap; confirm deployed/submitted digest identity; reconcile the report/release tag; verify the final video and submission links. No honest overall percentage can replace these gates.
