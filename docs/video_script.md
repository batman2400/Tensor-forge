# Video script (15 minutes, YouTube unlisted)

Team Fade. Model `v1.0.0-38ecb9bd`. Read the lines under **Say**. The lines under **Show** are what is on screen. Pause on the clicks. At a calm pace this is about 14 to 15 minutes.

Record in one take if you can. Upload as **Unlisted**, then open the link in a private window while logged out of YouTube and confirm it plays.

## Before you record

- Browser on the demo only: https://tensorforge-fade.southindia.cloudapp.azure.com/demo/
- A second tab on https://tensorforge-fade.southindia.cloudapp.azure.com/health
- The API key field is a password input. Paste the key once, off camera if you want, and never open `.env`, email, or the Azure portal.
- Do not show ticket text from the organizer dataset. The six example buttons on the demo are safe to read.
- The number to say for fusion accuracy is development-validation macro-F1 **0.802**. Fusion settings were fit on out-of-fold predictions and labels from all 4,800 tickets, including validation, so this is not an untouched holdout. If `experiments.csv` or an in-sample check is on screen, say that those higher figures are not this score.
- Optional upload: `docs/demo_tickets.csv` (three synthetic tickets). Do not upload `data/`.

## 0:00 — Open

**Show:** the demo page, already loaded, key field empty.

**Say:** This is TensorForge, team Fade, for the Phase 2 support-ticket task. RideEat is a fictional ride-hailing and food-delivery platform. A customer writes in by email, chat, or a call transcript, and the service has to classify the ticket and route it to a team.

The scoring criteria are correctness, schema compliance, and robustness. Latency is not scored. What the judges asked to see is a service that follows their API, and a model we trained and validated ourselves. Classification is our own models. No language-model API is called when a ticket comes in.

## 1:00 — What one ticket returns

**Show:** stay on the demo. Point at the empty decision panel.

**Say:** For each ticket the service returns five things. A primary category, one of eleven. The team that category always maps to, from the organizers' fixed table, so the model does not invent a team. An optional secondary category, when the ticket really has two issues, otherwise null. Whether it is urgent. And a confidence for the primary category only.

If that confidence is below 0.5, `needs_human_review` is true. Spam is never urgent, and spam never has a secondary category. The secondary category is never the same as the primary.

## 2:00 — The data

**Show:** nothing from `data/`. You can leave the demo up, or show `ml/reports/eda_findings.md` scrolled to the counts only.

**Say:** The labelled set is 4,800 tickets: 4,000 for training and 800 held out for validation. There are eleven categories. The largest is payment and refund, about 18 percent. Spam is about 3 percent. Because the classes are unbalanced, the headline metric is macro-F1, the unweighted average of the per-class F1 scores. Accuracy alone would flatter a model that always says payment and refund. That majority-class floor scores about 0.03 macro-F1 on the validation set.

Languages follow the brief: English, Singlish, Sinhala, Tamil, a little Tanglish, and a little mixed. Language is not an input feature. The API does not send it, and the model is not allowed to use the ticket id either. It only sees the channel, the subject, and the text.

Subjects exist on email. Chat and call transcripts do not have them. Most texts are short, a few hundred characters, even though the API allows up to 10,000. About 10 percent of tickets are urgent, and about 10 percent have a real secondary issue. A late-food refund is a delivery delay, with payment and refund as the secondary. A missing-food refund is an order missing or wrong, again with refund as the secondary.

One warning from the exploration: a lot of tickets share an opening template. So a very high score on a random split can mean the model memorized wording that appears on both sides. The classical and encoder-only numbers below are the 800 validation tickets. The fusion number is not an untouched holdout, because its settings saw those labels. Do not quote a score computed after the final encoder was fit on all 4,800.

## 4:00 — How the model was trained

**Show:** `ml/reports/experiments.csv` in the editor, or the metrics page later. If the CSV is visible, point at the `stage_a_macro_f1` column, not the `cv_macro_f1` column.

**Say:** Training happened off the server. The laptop and a Kaggle GPU produced frozen files. The running service only loads those files.

We started with a floor, then classical models. The input is a TF-IDF vector: word n-grams plus character n-grams, so spelling variation in Singlish and Tanglish still produces a vector. We compared logistic regression, linear SVM, SGD, naive Bayes, and a small gradient-boosted model, all with the same validation split and the same metric code.

The linear SVM with word and character n-grams won that search. On the 800 validation tickets its macro-F1 is 0.656. The others were lower. LightGBM on the same features was about 0.47, so we did not keep it. That SVM is still in the system. It is called `svc_word_char`.

The second model is a fine-tune of `intfloat/multilingual-e5-small`, which is an MIT-licensed encoder, not a chat model. We trained a classification head on our labels, on a Kaggle T4, with early stopping. Fit on the 4,000 training tickets and scored on the 800, its macro-F1 is 0.786. It is stronger on the primary category. On its own it was weaker on the secondary label, so we did not ship it alone.

Fusion is a weighted average of the two probability vectors. The weights and the temperatures were fit on out-of-fold predictions and labels across all 4,800 records, including the validation tickets, and the fused model was then scored on the validation set. The encoder gets three quarters of the vote on category and on secondary, and half the vote on urgency. That development-validation macro-F1 is **0.802**. Urgent F1 is 0.91. Secondary exact match is 0.96. The gain over the SVM alone is about 0.15 macro-F1. It is not an untouched holdout score.

After that measurement we fit the encoder one last time on all 4,800 tickets, for 4 epochs, and exported it to int8 ONNX. That file is what the server loads. A score of that file on the same 800 tickets is not a held-out result, because those tickets were inside the final fit. If you see a figure near 0.99 in the experiment log, that is cross-validation on overlapping templates, or that final fit. It is not the number for this project. The number to quote is the development-validation macro-F1, 0.802.

The int8 graph was checked against the floating-point graph. On a validation sample the fused labels matched, even though a few raw probabilities move.

## 7:30 — Architecture

**Show:** a simple sketch is enough: browser and judges, then Caddy, then the container. Or scroll `docs/ARCHITECTURE.md` section 3 for a few seconds. Do not open a terminal that has secrets in the environment.

**Say:** The live host is an Azure VM in South India. Caddy terminates HTTPS and proxies to one container on localhost. The container is the product. It starts with:

`docker run -p 8000:8000 -e API_KEY=<key> tensorforge:verify`

The key comes only from that environment variable. If it is unset, prediction returns 401. The image has the weights inside it, and it does not need a network at runtime. We have started it with networking disabled and it still served `/health` and a prediction.

Inside the process, one worker handles requests. A background worker runs large jobs so a 5,000-ticket submit can return immediately. Job state is a local SQLite file on a volume, so a poll from another request can see it. If the process restarts in the middle of a job, that job is marked failed with error code `interrupted`. Results are kept for 12 hours.

Auth is checked before the body. A missing or wrong key is 401, then a bad content type is 415, an oversized body is 413, broken JSON is 400, and a schema violation is 422. Both `X-API-Key` and `Authorization: Bearer` work. Unknown routes return JSON, not an HTML page.

There is one product rule on top of the model, for email. If the subject and the body name different issues, the body wins. A safety issue on either side wins. A vague body, such as "please help", follows the subject. We measured this on the validation set. It did not raise the validation macro-F1, so we do not claim it as an accuracy improvement. It is there so a one-line subject is not ignored when the body is empty of detail.

## 9:30 — Live demo

**Show:** the demo. Paste the key into the password field before this section, or paste it now without reading it aloud.

**Say:** This is the hosted demo. The key stays in this tab's session storage and is sent only as a header. It is not in the page source.

**Show:** click **English**. Click Predict. Wait for the panel.

**Say:** This sample is an English email: a customer charged twice for a ride. The panel shows the category, the team from the fixed map, urgency, secondary, confidence, and the model version. The version on this deployment is `v1.0.0-38ecb9bd`. Describe the panel you actually see. If confidence is under one half, point at the human-review flag.

**Show:** click **Romanized Sinhala**, then Predict. Then **Sinhala**, then Predict. Then **Tamil** or **Tanglish**, then Predict.

**Say:** Same service, no language field. Singlish, Sinhala, and Tamil go through the same encoder and the same TF-IDF vector. The multilingual encoder is why we added it: the classical model was weak on Sinhala and Tamil, and strong on Singlish because of shared templates. After fusion, English and Tamil moved up, Sinhala improved but is still the weakest slice, and Singlish came down from the classical score. The overall macro-F1 is higher. We did not win every language.

**Show:** click **Mixed**, then Predict.

**Say:** This one mentions a login code and a wrong receipt. The model has to pick a primary issue. If it also fills a secondary category, that is the second issue, and it must be a different class.

**Show:** if you prepared a 3-row CSV, drop it. Otherwise skip to the metrics page and say the next paragraph without the upload.

**Say:** A CSV with up to 100 rows uses the synchronous batch endpoint. A larger file, up to 5,000 rows, submits a job, polls status, and then loads the results. On this VM a 5,000-ticket job completed in 117 seconds, and the poll latency stayed under a second. The limit is 30 minutes. Locally, the same image under a 2 CPU and 4 GB cap finished that job in 72 seconds.

## 12:15 — Metrics page

**Show:** open Metrics. Point at the summary numbers, then the per-class table, then language, then the confusion matrix, then the reliability diagram. About 20 seconds on each.

**Say:** These charts are Stage A development-validation numbers. The base models were trained on 4,000 tickets and scored on the 800 validation tickets. Fusion macro-F1 is 0.802, and those fusion settings used validation labels, so this is not an untouched holdout. The per-class table is where the remaining errors sit: food and money tickets still get confused with each other, especially a missing order versus a lost item versus a refund. Ride problems are the main source of false safety predictions.

The reliability diagram compares confidence to accuracy. Expected calibration error on this split is about 0.06. The high bins are close to the diagonal. The middle bins are thinner and less perfect, which is why a confidence under 0.5 asks for a person.

## 13:30 — What we would not claim

**Show:** GitHub repository `batman2400/Tensor-forge`, the release `v1.0.1`, for a few seconds. Then back to `/health` in the browser, so `model_version` is visible. Do not scroll into `.env` or workflow logs.

**Say:** The repository has the service, the training scripts, the tests, the Dockerfile, and the frozen weights. The encoder file is about 118 megabytes, so it is stored with Git LFS. A clone needs Git LFS or the Docker build fails its own self-test. The organizer dataset is not in the repository. The API key is not in the repository.

The release tag for this submission is `v1.0.1`. The older `v1.0.0` tag is an earlier commit and is not this build. The model version string is `v1.0.0-38ecb9bd`, which is what `/health` returns.

Limits, said plainly. Sinhala is still behind English. The secondary head only emits the five secondary classes that actually appear in the data. The final encoder saw every training and validation ticket, so we do not treat a re-score of those tickets as evidence. And we did not wrap a general-purpose chat model around the labels.

## 14:30 — Close

**Show:** the demo decision panel, model version visible.

**Say:** Team Fade. A TF-IDF SVM plus a fine-tuned multilingual encoder, fused and exported to int8, served behind the organizers' contract on Azure. Development-validation macro-F1 0.802. Thank you.

## If you run long

Cut in this order, and keep the training section intact:

1. The mixed-language sample.
2. The CSV upload.
3. The status-code sentence in the architecture section.
4. The GitHub shot. Say the repository URL out loud instead.

## If you run short

Add one sentence after the SVM result: character n-grams are there because a word-only logistic regression scored about 0.50 macro-F1, and adding character n-grams lifted the same model to about 0.60 before the SVM search.
