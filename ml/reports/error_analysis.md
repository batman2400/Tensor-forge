# Error analysis (classical v0, Stage A)

Fit `svc_word_char` on train, score the 800 validation tickets, thresholds 0.5,
post-processing rules on. This is the same setup as the day-1 serving baseline.
Counts only: no ticket text.

Stage A accuracy 0.6338, macro-F1 0.6558,
urgent F1 0.8295 (precision 0.7604,
recall 0.9125), secondary exact 0.9550.
Urgent false positives: 23 / 800 (0.029).

## Where category errors concentrate

Macro-F1 is pulled down by the food-and-money cluster. `order_missing_wrong` is the
weak class. `lost_item` looks better on recall than precision because other classes
fall into it (62 validation tickets predicted `lost_item` with a different gold label).
Ride tickets are the main source of false `safety_conduct`
(25 non-safety tickets predicted as safety).

| category | F1 | support |
| --- | --- | --- |
| order_missing_wrong | 0.322 | 96 |
| lost_item | 0.489 | 40 |
| safety_conduct | 0.575 | 32 |
| payment_refund | 0.587 | 144 |
| food_quality | 0.588 | 56 |
| app_technical | 0.639 | 64 |
| general_inquiry | 0.680 | 40 |
| delivery_delay | 0.716 | 120 |
| ride_trip_issue | 0.792 | 112 |
| account_promo | 0.826 | 72 |
| spam_irrelevant | 1.000 | 24 |

Largest gold → predicted pairs:

| gold | predicted | count |
| --- | --- | --- |
| order_missing_wrong | lost_item | 26 |
| payment_refund | lost_item | 25 |
| order_missing_wrong | payment_refund | 25 |
| delivery_delay | payment_refund | 19 |
| ride_trip_issue | safety_conduct | 19 |
| app_technical | order_missing_wrong | 18 |
| food_quality | order_missing_wrong | 15 |
| payment_refund | general_inquiry | 15 |

## Labelling-rule slices

Accuracy is the primary category on that slice. Secondary exact is the full pair,
including a correct null. Urgent false-positive rate is predicted urgent when the
gold label is not. Mistake lists are the predicted category when the primary is wrong.

| slice | n | accuracy | secondary exact | urgent FP rate | top wrong predictions |
| --- | --- | --- | --- | --- | --- |
| safety_conduct | 32 | 0.719 | 0.875 | 0.062 | lost_item 5, payment_refund 3, ride_trip_issue 1 |
| late_food_refund | 9 | 0.778 | 0.778 | 0.000 | order_missing_wrong 1, safety_conduct 1 |
| missing_food_refund | 8 | 0.375 | 0.625 | 0.000 | delivery_delay 2, payment_refund 2, lost_item 1 |
| lost_item | 40 | 0.825 | 0.875 | 0.025 | delivery_delay 2, ride_trip_issue 2, payment_refund 1 |
| account_promo | 72 | 0.792 | 0.931 | 0.000 | payment_refund 7, general_inquiry 4, ride_trip_issue 2 |
| app_technical | 64 | 0.484 | 1.000 | 0.047 | order_missing_wrong 18, payment_refund 6, account_promo 5 |
| general_inquiry | 40 | 0.850 | 1.000 | 0.050 | payment_refund 3, delivery_delay 1, lost_item 1 |
| nonurgent_safety | 8 | 1.000 | 1.000 | 0.250 | none |
| spam_irrelevant | 24 | 1.000 | 1.000 | 0.000 | none |
| injection_text | 22 | 0.773 | 1.000 | 0.000 | delivery_delay 3, lost_item 1, ride_trip_issue 1 |
| resolved_wording | 55 | 0.527 | 0.873 | 0.073 | payment_refund 15, safety_conduct 5, delivery_delay 2 |
| quoted_correspondence | 17 | 0.588 | 0.824 | 0.059 | safety_conduct 4, payment_refund 2, delivery_delay 1 |

OTP/login versus post-login, measured as the two-way confusion: 0 `account_promo` predicted `app_technical`, 5 `app_technical` predicted `account_promo`.
Late-food refunds are `delivery_delay` with secondary `payment_refund`.
Missing-food refunds are `order_missing_wrong` with secondary `payment_refund`.
Non-urgent safety is the measurable stand-in for a resolved historical safety complaint:
the gold label is safety and not urgent. Spam is a separate row and should stay exact,
because the serving rule clears urgency and secondary after a spam primary.

Injection, resolved wording, and quoted correspondence use the same surface patterns
as the EDA counts. They are slices, not keyword features. A higher urgent false-positive
rate on the injection slice than the overall rate means the model is treating the
instruction text as a signal.

## Language and channel

Singlish is much easier than Sinhala or English on this split. Shared opening templates
are a known property of the pool, so a high Singlish score is not a reason to add
`language` as a feature. The model still does not receive `language` or `ticket_id`.

| language | n | accuracy | macro-F1 |
| --- | --- | --- | --- |
| en | 280 | 0.550 | 0.592 |
| mixed | 20 | 0.600 | 0.462 |
| si | 160 | 0.450 | 0.529 |
| singlish | 200 | 0.930 | 0.938 |
| ta | 120 | 0.592 | 0.558 |
| tanglish | 20 | 0.600 | 0.524 |

| channel | n | accuracy | macro-F1 |
| --- | --- | --- | --- |
| call_transcript | 200 | 0.610 | 0.597 |
| chat | 360 | 0.664 | 0.653 |
| email | 240 | 0.608 | 0.664 |

Email has subjects; chat and calls do not. That is already in `build_input`.

## What this asks of the v1 candidate

- Lift `order_missing_wrong` away from `lost_item` and `payment_refund` without giving
  those classes away. That is the macro-F1 lever.
- Stop calling ordinary ride issues `safety_conduct`.
- Keep secondary exact near the v0 level (0.955). It is already a strength.
- Urgency is high-recall and lower-precision. Non-urgent safety (2 of 8 predicted urgent)
  and resolved wording (urgent false-positive rate 0.073 versus 0.029 overall) are where
  false urgency shows up. The injection slice had no urgent false positives, so this split
  does not support a special injection parser.
- `app_technical` is often predicted as `order_missing_wrong` (18 of 64). The OTP versus
  post-login swap is small: 0 promo tickets called app issues, 5 app tickets called promo.
- Do not add a hard keyword rule for refunds, OTP, or resolved mail. The native-script
  rows will not match an English keyword, and these slices are not precise enough.
