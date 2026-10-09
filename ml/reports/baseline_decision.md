# Classical baseline v0 decision

Stage A (fit on train, score on validation) is the number to quote.
Random 5-fold scores on the pooled 4,800 tickets are much higher because tickets share
opening templates across folds, and the organizer split has no exact text overlap.
OOF is only a ranking check. Thresholds are 0.5 (not tuned). Review threshold is 0.5.

| config | stage A acc | stage A macro-F1 | stage A urgent F1 | stage A secondary exact | CV macro-F1 |
| --- | --- | --- | --- | --- | --- |
| logreg_word | 0.4600 | 0.5014 | 0.8252 | 0.9525 | 0.9639 |
| logreg_word_char | 0.5787 | 0.6037 | 0.8725 | 0.9513 | 0.9755 |
| svc_word_char | 0.6338 | 0.6558 | 0.8295 | 0.9550 | 0.9877 |

Chose `svc_word_char` on Stage A macro-F1 0.6558 (accuracy 0.6338), 0.0521 above `logreg_word_char`. OOF macro-F1 ranks it the same way (0.9877 vs 0.9755), but those OOF scores are not the accuracy to publish. Stage A urgent F1 is 0.8295, below `logreg_word_char` at 0.8725. Category macro-F1 is the selection key; the urgency threshold stays at 0.5 for day 2. The serving artifact is this configuration refit on all 4,800 tickets, version v0.1.0. Thresholds are deliberately untuned at 0.5 so day-2 error analysis can move them without mixing that choice into v0.
