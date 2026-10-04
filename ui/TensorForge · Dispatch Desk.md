TensorForgeDispatch Desk

model loaded · v2.0 · cpu

## 01Single ticket

POST /predict

Incoming ticket

Nothing on the desk yet.\
Pick a sample or paste a ticket.

## 02Batch

POST /predict/batch · 1–100 tickets, one per line

[ ] Urgent only

| # | Ticket | Category | Team | Secondary | Urgent | Conf. |
| --- | --- | --- | --- | --- | --- | --- |

## 03Async job

POST /batch/jobs · up to 5,000 tickets · polled until complete

Upload tickets (.csv / .jsonl)

Drop a file here — demo uses 1,200 sample tickets

job: —idle0 / 1,200

Submit a job to begin.

Mock responses — swap `predict()` for fetch() calls to your FastAPI endpoints. Category and team names are placeholders.