# Classical v1 candidate

Stage A (fit on train, score on validation) is the number to quote.
The search below did not replace `svc_word_char`. That configuration, thresholds 0.5,
remains the classical candidate and the live serving file `artifacts/classical.joblib` (v0.1.0).
No second artifact was written. Five-fold CV was not re-run: those scores are inflated by shared templates.

| config | stage A acc | stage A macro-F1 | stage A urgent F1 | stage A secondary exact |
| --- | --- | --- | --- | --- |
| svc_word_char (v0) | 0.6338 | 0.6558 | 0.8295 | 0.9550 |
| nb_word_char | 0.5525 | 0.5490 | 0.6667 | 0.7350 |
| sgd_word_char | 0.6112 | 0.6356 | 0.8707 | 0.9525 |
| logreg_c4_word_char | 0.6050 | 0.6282 | 0.8630 | 0.9525 |
| logreg_c025_word_char | 0.5413 | 0.5634 | 0.8366 | 0.9513 |
| sgd_char36 | 0.6162 | 0.6403 | 0.8571 | 0.9525 |
| nb_char36 | 0.5563 | 0.5520 | 0.6798 | 0.7412 |
| svc_mindf1 | 0.6388 | 0.6630 | 0.8111 | 0.9550 |
| svc_word3 | 0.6325 | 0.6533 | 0.8114 | 0.9550 |
| lgbm_word_char | 0.4663 | 0.4747 | 0.8116 | 0.9475 |

`lgbm_word_char` (120 trees, class-balanced, on the same word+char TF-IDF) scores Stage A macro-F1 0.4747, 0.181 below v0. `order_missing_wrong` F1 rises only from 0.322 to 0.347, Sinhala accuracy falls from 0.450 to 0.350, and urgent F1 falls from 0.8295 to 0.8116. It does not enter the candidate.

`svc_mindf1` is the only run above v0 on macro-F1, by 0.0071.
It does not fix the error-analysis target: `order_missing_wrong` F1 falls from 0.322 to 0.308,
Sinhala accuracy does not rise (0.450 to 0.444), and urgent F1 falls from 0.8295 to 0.8111.
The gain sits in `general_inquiry`, `safety_conduct`, and `app_technical`.
That is too mixed to change the candidate. `svc_word3` is slightly under v0.

`sgd_word_char` has the best urgent F1 in the table (0.8707) and keeps secondary exact at 0.9525,
but its macro-F1 is 0.6356. Category macro-F1 stays the selection key, so SGD is not the candidate.
It is the urgency reference if thresholds are revisited on a split that does not share templates.

A 20% holdout inside train cannot set those thresholds.
On `svc_word_char` that holdout scored secondary exact 1.000 and urgent F1 0.994,
because train rows share opening templates. The grid therefore stayed at 0.5.
Validation urgent F1 remains 0.8295. Do not publish the holdout figure.

The API still serves v0.1.0. Gate G1 is still open: the encoder leads on category macro-F1
(0.786) and urgent F1 (0.908) and trails on secondary exact (0.721 versus 0.955).
