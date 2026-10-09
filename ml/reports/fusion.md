# Fusion

Weights are the encoder's share of a weighted average with `svc_word_char`.
They are fit on out-of-fold probabilities and labels across all 4,800 records, including validation. Stage A below is tuning-exposed validation, not an untouched holdout estimate.
The serving manifest fuses this with the final encoder (`artifacts/manifest.json`).

- Category encoder weight 0.75, temperature 0.5.
- Secondary encoder weight 0.75, temperature 2.0.
- Urgent encoder weight 0.5, temperature 1.0.
- Thresholds secondary 0.5, urgent 0.5, review 0.5.

| setup | stage A acc | stage A macro-F1 | stage A urgent F1 | stage A secondary exact |
| --- | --- | --- | --- | --- |
| svc_word_char | 0.6338 | 0.6558 | 0.8295 | 0.9550 |
| fusion | 0.7975 | 0.8022 | 0.9080 | 0.9637 |

Stage A macro-F1 changes by +0.1463 versus `svc_word_char`. Laptop ONNX latency is in `onnx_latency.md`. Gate G1 passed; see `g1.md`. The reliability diagram is in `calibration.md`.
