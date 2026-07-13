# FP32 → INT8 quantization results — measured 2026-07-02

**Hardware:** Intel Core i5-1135G7 (Tiger Lake, 4C/8T), 16GB RAM, CPU only, Windows 11.
OpenVINO `OPTIMIZATION_CAPABILITIES` reports `INT8` → **VNNI/DL Boost present**.
**Config:** imgsz=640, batch=1, `OMP_NUM_THREADS=4`, `torch.set_num_threads(4)`;
OpenVINO (TBB) and ONNX Runtime use their own default thread pools (physical cores).
ultralytics 8.4.84 / openvino 2026.2.1 / onnxruntime 1.27.0.
**Method:** end-to-end `predict()` wall time, 20 warmup + 150–200 warm runs, all artifacts
**interleaved round-robin** so every model sees identical instantaneous machine conditions
(see `model/benchmark.py` header for why).

## Headline: OpenVINO FP32 vs OpenVINO INT8 (same backend — the honest delta)

| Condition | FP32 p50 | INT8 p50 | wall-latency reduction |
|---|---|---|---|
| Sustained (record run, 150 interleaved runs) | 30.8 ms | 27.8 ms | **−9.7%** |
| Idle-window run (60 interleaved runs) | 22.6 ms | 18.9 ms | **−16.4%** |
| Inference stage only (idle window, ultralytics per-stage timing) | 13.9 ms | 11.0 ms | **−20.9%** |

| | FP32 | INT8 | delta |
|---|---|---|---|
| mAP@0.5 (test, 1234 imgs) | 0.7866 | 0.7880 | +0.0014 |
| mAP@0.5:0.95 (test) | 0.4996 | 0.4945 | **−0.0051 (−1.0% relative)** |
| model size | 10.6 MB | 3.2 MB | −70% |

The wall-time gain is smaller than the inference-stage gain because ~5–6 ms of each request
is decode/letterbox/postprocess that quantization cannot touch.

## Full record table (sustained conditions, interleaved, 150 runs)

| Backend | Precision | p50 (ms) | p95 (ms) | p99 (ms) | img/s | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|---|
| PyTorch | FP32 | 102.0 | 133.4 | 331.7 | 9.0 | 0.7905 | 0.4964 |
| ONNX RT | FP32 | 75.8 | 88.1 | 162.7 | 13.2 | 0.7869 | 0.4991 |
| ONNX RT | INT8 | 145.9 | 224.5 | 385.2 | 6.1 | — * | — * |
| OpenVINO | FP32 | 30.8 | 90.3 | 121.4 | 26.8 | 0.7866 | 0.4996 |
| OpenVINO | INT8 | **27.8** | 93.9 | 133.0 | 25.3 | 0.7880 | 0.4945 |

\* not evaluated: ONNX INT8 was 2× slower than ONNX FP32, so it was dropped from consideration.

## Honest findings, including the negative ones

1. **OpenVINO INT8 is the winner** — fastest p50 and 3.3× smaller, at a cost of 0.005 mAP@0.5:0.95.
2. **ONNX Runtime INT8 (QDQ static quantization) is ~2× SLOWER than ONNX FP32** on this
   machine despite VNNI. QDQ dequantize/quantize node overhead dominates on a 2.7M-param
   nano model. INT8 ≠ automatic speedup; measure per backend.
3. **The OpenVINO backend switch (ONNX RT 75.8 → OpenVINO 30.8 ms) is a bigger win than
   quantization itself (30.8 → 27.8 ms).** These are deliberately reported separately:
   comparing ONNX-FP32 to OpenVINO-INT8 and calling it a "quantization speedup" would
   conflate backend and precision effects.
4. **Measurement traps found and worked around** (documented for reproducibility):
   - Windows 11 throttles background-launched processes (EcoQoS): p50 inflated up to 5×.
     All record runs are foreground.
   - This laptop CPU thermally settles ~35% slower under sustained load than in short
     bursts (idle-window vs sustained rows above). Deltas are stable across both states;
     absolute numbers depend on thermal state. p95/p99 tails include interference from
     other desktop processes (this is a shared laptop, not a dedicated bench box).
   - Sequential per-model benchmarking is unfair on thermally-limited hardware; the
     benchmark interleaves models per-image instead.

Deploy-target (HF Spaces) numbers to be added in Phase 7 — server CPUs won't have the
laptop thermal/interference caveats.
