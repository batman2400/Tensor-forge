# WP7 service measurements

Base URL `http://127.0.0.1:8001`. Model `v1.0.0-38ecb9bd`.
These numbers are from the machine that ran the bench, not a claim about the Azure VM
until that host is named in the notes.

| check | result | target |
| --- | --- | --- |
| `/predict` p50 / p95 | 27.7 ms / 48.3 ms | p95 under 1 s |
| 100-ticket batch | 1.93 s | under 30 s |
| 2000-ticket job | 53.3 s, submit 55 ms, health p95 27.7 ms | job under 30 min, submit under 5 s, health under 1 s |
| memory during 2000-ticket job | 558 MiB | under about 2 GB |
| 5000-ticket job | 128.4 s, submit 54 ms, health p95 26.4 ms | job under 30 min, submit under 5 s, health under 1 s |
| memory during 5000-ticket job | 563 MiB | under about 2 GB |
| startup to HTTP 200 | 4.58 s | under 120 s |
| fuzz | 11 cases, 0 server errors or non-JSON | no 5xx |
| 20 parallel `/predict` | 0 server errors | no 5xx |

## Notes

- Container tf-wp7: docker run --cpus 2 --memory 4g --memory-swap 4g, image tensorforge:dev (994MB), model v1.0.0-38ecb9bd. Cores are this laptop, not the Azure B2als_v2.
- A second start with --network none reached /health 200 in 3.26 s. docker history of the image has no tf2_ key and no baked API_KEY.
- Startup time includes creating the container. The unlimited container named tensorforge was left running and was idle.
