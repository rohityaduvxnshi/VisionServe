# VisionServe

CPU-only computer-vision inference service for **PPE / safety-gear detection** (helmet,
vest, gloves, goggles, mask, safety shoe), with a measured **FP32 → INT8 optimization
story**: export a YOLO nano detector to OpenVINO, quantize to INT8 with calibration, and
prove the latency/accuracy trade-off honestly — same backend, same machine, same thread
config, held-out test split.

**Live demo:** `MEASURE` (deployed URL pending — see Deploy) · **Demo GIF:** `MEASURE`

FastAPI backend + thin React frontend that draws detections and shows a live
**FP32 ↔ INT8 latency toggle**.

## Headline results (local: Intel i5-1135G7, 4C/8T, VNNI, 16GB, Windows 11)

Same backend (OpenVINO), same machine, interleaved measurement, imgsz=640, batch=1,
`OMP_NUM_THREADS=4`. Full method + table in [eval/quantization.md](eval/quantization.md).

| Backend | Precision | p50 (ms) | img/s | mAP@0.5 | mAP@0.5:0.95 | size |
|---|---|---|---|---|---|---|
| PyTorch | FP32 | 102.0 | 9.0 | 0.7905 | 0.4964 | 5.4 MB |
| ONNX RT | FP32 | 75.8 | 13.2 | 0.7869 | 0.4991 | 10.5 MB |
| ONNX RT | INT8 | 145.9 | 6.1 | — | — | 3.0 MB |
| OpenVINO | FP32 | 30.8 | 26.8 | 0.7866 | 0.4996 | 10.6 MB |
| **OpenVINO** | **INT8** | **27.8** | 25.3 | 0.7877 | **0.4946** | **3.2 MB** |

- **Quantization (same backend):** OpenVINO INT8 vs FP32 = **−10% wall p50 sustained /
  −16% in idle-window runs / −21% inference-stage only**, **−70% model size**, at
  **−0.005 mAP@0.5:0.95 (−1% relative)** on the 1234-image held-out test split.
- **Backend choice** (reported separately — not a quantization win): ONNX Runtime →
  OpenVINO on this Intel CPU is 75.8 → 30.8 ms.
- **Honest negative result:** ONNX Runtime INT8 (QDQ static quant) was ~2× *slower* than
  ONNX FP32 on this nano model despite VNNI. INT8 is not an automatic speedup.
- Deployed target (HF Spaces 2 vCPU): `MEASURE` (re-run `model/benchmark.py` there).

## Evaluation (held-out test split: 1234 images / 1782 instances)

Full report with per-class P/R and confusion matrices: [eval/README.md](eval/README.md).
Key read: **zero inter-class confusion** — all model error is missed *small* objects
(gloves 38%, safety shoes 36%, masks 30% missed) and background false positives. Vest
(0.93 mAP@0.5) and goggles (0.94) are the strongest classes.

## Architecture

```
React SPA (Vercel) ──HTTPS──► FastAPI (HF Spaces, Docker, CPU)
  upload → canvas boxes         loads OpenVINO INT8 (default) + FP32 (toggle)
  FP32/INT8 latency toggle      Ultralytics wrapper: letterbox + decode + boxes
                                in original-image pixel space
offline: model/export.py (ONNX, OpenVINO IR, INT8+calibration) → model/benchmark.py → eval/
```

### API contract (the seam — frontend is a thin REST client)

```
POST /predict            multipart field "image"; query: model=int8|fp32, conf (0.25)
  → { "model": "int8", "inference_ms": 27.8, "image": {"width": w, "height": h},
      "detections": [{ "class_id", "class_name", "confidence",
                       "box": {"x1","y1","x2","y2"} }] }   # absolute px, original image
POST /predict/batch      multipart "images" (≤16) → { "results": [ ... ] }
GET  /health             → { "status": "ok" }
GET  /model/info         → { "format", "classes", "input_size", "version" }
```

## Model & data

- **Model:** pre-trained YOLOv8-nano PPE detector
  ([Tanishjain9/yolov8n-ppe-detection-6classes](https://huggingface.co/Tanishjain9/yolov8n-ppe-detection-6classes), MIT).
  **I did not train this model** — this project's focus is the serving/optimization
  pipeline. A one-time YOLO26n fine-tune on free Colab GPU is provided in
  [model/train_colab.ipynb](model/train_colab.ipynb) and drops in without pipeline changes.
- **Dataset:** [PPE Detection_DATA v3](https://universe.roboflow.com/tanish-y7iqo/ppe-detection_data-adcya)
  (Roboflow, CC BY 4.0), YOLO format, 8774/2070/1234 train/val/test. Measured test mAP
  matches the model card's self-reported numbers, confirming the model↔dataset pairing.

## Reproduce

```powershell
py -3.12 -m venv .venv && .venv\Scripts\pip install -r requirements.txt
# dataset (667MB) → data/ppe/  (see data/ppe.yaml header for URL + one-time datasets_dir setting)
.venv\Scripts\python model\export.py                       # ONNX + OpenVINO FP32/INT8
.venv\Scripts\python model\benchmark.py --models ...       # foreground, idle machine!
.venv\Scripts\python -m uvicorn server.app:app --port 8000 # API
cd web && npm install && npm run dev                       # UI (VITE_API_URL in .env)
pytest server\test_api.py                                  # contract tests
```

Docker: `docker build -f server/Dockerfile -t visionserve .` (CPU-only wheels; HF Spaces
builds this same Dockerfile via [deploy/stage_hf_space.ps1](deploy/stage_hf_space.ps1)).

## Key decisions & trade-offs

- **Same-backend comparison only.** OpenVINO-FP32 vs OpenVINO-INT8 is the quantization
  claim; the ONNX→OpenVINO backend win is reported separately. Conflating them is the
  most common way these numbers lie.
- **Inference via the Ultralytics wrapper** on exported models — letterbox preprocessing
  and coordinate back-mapping are correct by construction rather than hand-rolled.
- **Both served models are OpenVINO** so the UI toggle demos quantization, not backends.
- **Interleaved benchmarking** (all models round-robin per image): on a thermally-limited
  laptop, sequential per-model runs are unfair; interleaving keeps the delta trustworthy.
- **No response cache** — it would fake the latency demo. No micro-batching — complexity
  without payoff at demo scale.

## What didn't work / limitations

- **ONNX Runtime INT8 was 2× slower than FP32** (QDQ overhead on a 2.7M-param model).
- **INT8 speedup is real but modest (~10–20%)** on this nano model: it is memory/overhead
  bound, and ~5 ms/request is preprocessing that quantization can't touch. Bigger models
  quantize better. The INT8 win requires VNNI — on CPUs without it, INT8 can be slower
  (measure per target; HF Spaces CPU: `MEASURE`).
- **Small-object recall** is the model's real weakness (gloves/shoes/masks) — see eval.
- **Laptop benchmarking noise:** Windows EcoQoS throttles background processes (up to 5×);
  thermal state shifts absolute numbers ~35% between burst and sustained load. Method and
  both states documented in eval/quantization.md.
- Single-image scope (no video/streams), HF Spaces cold start after ~48h idle,
  dataset split integrity relies on Roboflow's export.
- **Licenses:** Ultralytics is AGPL-3.0 → this repo is open source; don't reuse in closed
  commercial software without an Ultralytics enterprise license. Model weights MIT,
  dataset CC BY 4.0.

## AI-assistance transparency

Built with AI assistance (Claude) for scaffolding, API/UI code, and benchmark tooling.
The methodology decisions — same-backend comparison, interleaved benchmarking, calibration
dataset choice, diagnosing the EcoQoS/thermal measurement traps, and the confusion-matrix
analysis — are documented step-by-step in `eval/` and every number there was measured on
this machine, not generated. I can walk through and defend any of them.

## What I'd do next

Fine-tune YOLO26n (notebook provided) and re-run the identical pipeline; per-class
confidence thresholds; tiled inference for small-object recall; async inference queue if
concurrent load ever matters.
