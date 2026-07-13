# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CPU-only PPE-detection inference service (helmet, vest, gloves, goggles, mask, safety shoe)
whose whole point is a **measured FP32 → INT8 quantization story on OpenVINO**: same backend,
same machine, interleaved benchmarking, held-out test split. The model is a pre-trained
YOLOv8-nano (not trained here); the deliverable is the serving/optimization pipeline and the
honesty of its numbers. Full results and method: [README.md](README.md), [eval/](eval/).

## Commands

All Python runs through the repo venv (Python 3.12):

```powershell
.venv\Scripts\python -m pytest server\test_api.py -q        # API contract tests (needs data/ppe/test/images)
.venv\Scripts\python -m pytest server\test_api.py -q -k predict   # single test
.venv\Scripts\python -m uvicorn server.app:app --port 8000  # run API locally
.venv\Scripts\python model\export.py                        # re-export ONNX + OpenVINO FP32/INT8
.venv\Scripts\python model\benchmark.py --models model\ppe_yolov8n_openvino_model model\ppe_yolov8n_int8_openvino_model
cd web; npm run dev                                          # frontend (VITE_API_URL in web/.env)
cd web; npm run build                                        # tsc -b && vite build
cd web; npm run lint                                         # oxlint
docker build -f server/Dockerfile -t visionserve .           # CPU-only image (root build context)
deploy\stage_hf_space.ps1                                    # regenerate deploy/hf-space/ staging repo
```

**Benchmark honesty rules (do not violate):**
- Run `benchmark.py` in a **foreground shell on an idle machine** — Windows EcoQoS throttles
  background-launched processes up to 5×, silently corrupting latency numbers. mAP is unaffected.
- Latency comparisons are only valid **same-backend** (OpenVINO FP32 vs OpenVINO INT8).
  The ONNX→OpenVINO delta is a backend win and is reported separately in the README.
- The benchmark interleaves all models round-robin per image on purpose (thermal fairness);
  don't "simplify" it to sequential per-model loops.
- `benchmark.py` merges into `eval/benchmarks.json` per-field, so a `--no-map` rerun keeps stored mAP.

## Architecture

Two halves that meet at the exported OpenVINO model directories:

- **Offline pipeline:** `model/export.py` (PT → ONNX / OpenVINO IR / INT8 with NNCF calibration
  from `data/ppe.yaml`) → `model/benchmark.py` (interleaved latency + test-split mAP →
  `eval/benchmarks.json`) → hand-written analysis in `eval/*.md`.
- **Serving:** `server/app.py` (FastAPI routes, CORS, upload limits) is a thin shell over
  `server/inference.py`, which owns everything model-related: loads **both** OpenVINO models
  (FP32 + INT8) at lifespan startup, warms them up, and serves predictions under one global
  lock (Ultralytics predictors aren't thread-safe). Inference goes through the Ultralytics
  `YOLO` wrapper so letterboxing and coordinate back-mapping are correct by construction —
  boxes returned in **original-image pixel space**.
- **Frontend:** `web/` React 19 + Vite SPA, a deliberately thin REST client of the API contract
  in the README (`/predict`, `/predict/batch`, `/health`, `/model/info`). It draws canvas boxes
  and toggles `model=int8|fp32` to demo the latency delta live.
- **Deploy:** `deploy/stage_hf_space.ps1` copies server + model dirs into `deploy/hf-space/`
  (a standalone HF Spaces Docker repo, gitignored); frontend targets Vercel.

Server config is env-driven: `MODEL_DIR`, `DEFAULT_MODEL` (int8), `CONF_THRESHOLD` (0.25),
`ALLOWED_ORIGINS`. Class names are mixed-case from the dataset (`Gloves`, `Vest`, `goggles`,
`helmet`, `mask`, `safety_shoe`) — tests assert this exact list; don't normalize.

Intentional non-features: no response cache (would fake the latency demo), no micro-batching,
single-image scope (no video/streams).

## Version control

- Single `main` branch; only the initial commit exists so far — **all project work is currently
  uncommitted/untracked**. Commit in logical units when asked.
- Gitignored by design (large or regenerable): `data/*` (except `ppe.yaml`), `*.pt`, `*.onnx`,
  `model/*_openvino_model/`, `runs/`, `web/dist/`, `deploy/hf-space/`, `.venv/`, `.env`.
  Model artifacts are reproducible via `model/export.py`; the dataset (667 MB) re-downloads
  from Roboflow (URL in `data/ppe.yaml` header).
- Licenses constrain reuse: Ultralytics is AGPL-3.0 (repo must stay open source), weights MIT,
  dataset CC BY 4.0.

## Project status (maintain this section)

Done: export, benchmarks, eval writeups, FastAPI server + contract tests (7 passing),
React UI, Dockerfile, HF Space staging script.

Pending (the `MEASURE` placeholders in README.md):
- Deploy backend to HF Spaces and frontend to Vercel — **blocked: no `hf`/`vercel` CLI
  installed or authenticated on this machine** (checked 2026-07-13).
- Re-run `model/benchmark.py` on the HF Space (2 vCPU) and fill the README's deployed-target numbers.
- Record demo GIF; replace `MEASURE` placeholders with live URLs.
- Optional next steps listed in README: YOLO26n fine-tune via `model/train_colab.ipynb`,
  per-class conf thresholds, tiled inference for small objects.

Known wart: `deploy/hf-space/README.md` front-matter emoji is mojibake (`ðŸ¦º`) — PowerShell 5.1
encoding issue in `stage_hf_space.ps1`; fix before pushing the Space.
