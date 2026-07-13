# FP32 baseline — measured 2026-07-02

The reference every later number is judged against. Recorded **before** any quantization.

## Setup

- **Model:** `model/ppe_yolov8n.pt` — pre-trained YOLOv8-nano PPE detector
  ([Tanishjain9/yolov8n-ppe-detection-6classes](https://huggingface.co/Tanishjain9/yolov8n-ppe-detection-6classes), MIT).
  **Fallback path: we did not train this model** (spec's no-training option). The primary path
  (`model/train_colab.ipynb`, YOLO26n fine-tune) can replace it later without pipeline changes.
- **Dataset:** PPE Detection_DATA v3 (CC BY 4.0), 6 classes, held-out **test** split: 1234 images / 1782 instances.
- **Hardware:** Intel Core i5-1135G7 (Tiger Lake, 4C/8T), 16GB RAM, CPU only.
  OpenVINO reports `INT8` in `OPTIMIZATION_CAPABILITIES` → VNNI/DL Boost present.
- **Software:** Python 3.12.10, ultralytics 8.4.84, torch 2.12.1+cpu. `OMP_NUM_THREADS=4`, `torch.set_num_threads(4)`.
- **Method:** end-to-end `model.predict()` (preprocess + inference + postprocess), single image,
  imgsz=640, 20 warmup + 200 warm runs cycling over val images. mAP via `model.val(split="test")`.

## Latency (PyTorch FP32, CPU)

| p50 (ms) | p95 (ms) | p99 (ms) | throughput (img/s) |
|---|---|---|---|
| 64.3 | 69.4 | 76.7 | 15.45 |

## Accuracy (test split)

| Class | Images | Instances | P | R | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|---|
| **all** | 1234 | 1782 | 0.814 | 0.732 | **0.790** | **0.496** |
| Gloves | 164 | 298 | 0.702 | 0.562 | 0.633 | 0.454 |
| Vest | 245 | 472 | 0.831 | 0.939 | 0.930 | 0.732 |
| goggles | 126 | 139 | 0.893 | 0.904 | 0.953 | 0.491 |
| helmet | 284 | 597 | 0.879 | 0.794 | 0.889 | 0.575 |
| mask | 43 | 86 | 0.884 | 0.619 | 0.695 | 0.379 |
| safety_shoe | 94 | 190 | 0.696 | 0.574 | 0.642 | 0.347 |

Notes: measured test mAP (0.790/0.496) is consistent with the model card's self-reported val
numbers (~0.81/~0.53), confirming the model↔dataset pairing. Weakest classes are the small
objects (Gloves, safety_shoe, mask); Vest and goggles are strongest. Test split contains 373
background (no-object) images that count as negatives.
