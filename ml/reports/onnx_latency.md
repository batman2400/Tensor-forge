# ONNX export

Graph: `intfloat/multilingual-e5-small` with the three task heads.
The checkpoint is the final fit on all 4,800 tickets for 4 epochs, the stage-A best epoch. Batch 8 with 2 accumulation steps. Raw int8 probabilities can move by about 0.19, and category argmax still matched on 200/200. After fusion with the classical branch, category, secondary, urgent, and the review flag matched the fp32 graph on 200/200. That int8 file is the one in `artifacts/`.

- Checkpoint: `ml\runs\encoder_e5_final\final\model.pt`
- int8 file: `ml\runs\encoder_final_onnx\encoder.int8.onnx` (118089624 bytes)
- fp32 file: `ml\runs\encoder_final_onnx\encoder.fp32.onnx` (470262882 bytes)
- Padding to 256 max abs gap on 20 tickets: 0.000000
- Serving pad_to: None
- RSS after the int8 session loaded: 1741.0 MB
- Machine: 22 CPUs, Intel64 Family 6 Model 170 Stepping 4, GenuineIntel

| check | value |
| --- | --- |
| fp32 category max abs vs PyTorch | 5.160e-07 |
| int8 category max abs vs PyTorch | 0.190606 |
| int8 secondary max abs vs PyTorch | 0.206949 |
| int8 urgent max abs vs PyTorch | 0.144971 |
| int8 category argmax agreement | 1.0000 |

An argmax can change only when the top-two margin is within twice the int8 category error (0.3812). The smoke heads are undertrained, so many tickets sit in that band. The probability match is the parity check.

Latency is one ticket at a time, ONNX Runtime intra-op threads = 1.
This machine is not the 2 vCPU / 4 GB container.

| slice | n | p50 ms | p95 ms | max ms |
| --- | --- | --- | --- | --- |
| validation sample | 50 | 15.9 | 32.2 | 50.3 |
| 10k-char text | 20 | 61.9 | 64.0 | 65.0 |

The int8 file is 118.1 MB, above GitHub's 100 MB file limit, so it stays out of git and is copied into the image from the local artifacts directory. Parity below is against this checkpoint.
