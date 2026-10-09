# Calibration (Stage A fusion)

Weights and temperatures are the frozen settings in `artifacts/fusion.json`.
They were fit on out-of-fold probabilities. This table scores the train-only
models on the 800 validation tickets. The shipped encoder was fit again on
all 4,800 tickets, so a score of that file on validation is not this number.

Accuracy 0.7975, macro-F1 0.8022, ECE 0.0589, Brier 0.2925.

| bin | n | mean confidence | accuracy |
| --- | --- | --- | --- |
| 0.0–0.1 | 0 |  |  |
| 0.1–0.2 | 0 |  |  |
| 0.2–0.3 | 2 | 0.289 | 0.000 |
| 0.3–0.4 | 20 | 0.364 | 0.250 |
| 0.4–0.5 | 35 | 0.459 | 0.457 |
| 0.5–0.6 | 35 | 0.551 | 0.571 |
| 0.6–0.7 | 48 | 0.641 | 0.479 |
| 0.7–0.8 | 61 | 0.753 | 0.508 |
| 0.8–0.9 | 146 | 0.854 | 0.747 |
| 0.9–1.0 | 453 | 0.966 | 0.958 |

The diagram is `calibration.svg`. A bin on the diagonal is calibrated.
Leave-one-language-out is not in this table: that would retrain without
each language. These are slices of the same Stage A predictions.

| language | n | accuracy | macro-F1 |
| --- | --- | --- | --- |
| en | 280 | 0.868 | 0.880 |
| mixed | 20 | 0.800 | 0.711 |
| si | 160 | 0.650 | 0.624 |
| singlish | 200 | 0.775 | 0.774 |
| ta | 120 | 0.850 | 0.781 |
| tanglish | 20 | 0.900 | 0.883 |

| channel | n | accuracy | macro-F1 |
| --- | --- | --- | --- |
| call_transcript | 200 | 0.810 | 0.819 |
| chat | 360 | 0.775 | 0.778 |
| email | 240 | 0.821 | 0.838 |

| slice | n | accuracy | secondary exact | urgent FP rate |
| --- | --- | --- | --- | --- |
| safety_conduct | 32 | 0.906 | 0.969 | 0.062 |
| late_food_refund | 9 | 1.000 | 0.778 | 0.000 |
| missing_food_refund | 8 | 0.500 | 0.750 | 0.000 |
| lost_item | 40 | 0.700 | 0.900 | 0.000 |
| account_promo | 72 | 0.931 | 0.931 | 0.000 |
| app_technical | 64 | 0.625 | 1.000 | 0.000 |
| general_inquiry | 40 | 0.900 | 1.000 | 0.000 |
| nonurgent_safety | 8 | 1.000 | 1.000 | 0.250 |
| spam_irrelevant | 24 | 0.792 | 1.000 | 0.000 |
| injection_text | 22 | 0.727 | 1.000 | 0.000 |
| resolved_wording | 55 | 0.855 | 0.945 | 0.018 |
| quoted_correspondence | 17 | 0.941 | 0.941 | 0.000 |
