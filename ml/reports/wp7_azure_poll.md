# Azure poll check

A 2,000-ticket job against `https://tensorforge-fade.southindia.cloudapp.azure.com` while sampling `/health` and job status from this laptop. Timeout per sample was 5 seconds.

| check | result |
| --- | --- |
| job | succeeded, 2000 / 2000 |
| submit | 1.9 s |
| failed samples | 0 |
| `/health` p50 / p95 / max | 374 ms / 844 ms / 1602 ms |
| poll p50 / p95 / max | 378 ms / 761 ms / 1281 ms |

p95 is under 1 second. The slowest health sample was 1.6 seconds. An earlier 5,000-ticket run had one external connection time out; this run did not.

The demo is served at `/demo/` from the same container. The visitor pastes the key; it is kept in session storage and is not in the page.
