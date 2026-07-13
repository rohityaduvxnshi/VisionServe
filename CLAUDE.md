# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Update this file at the end of every phase and tag git accordingly.**

## Header

- **Project:** VisionServe — CPU-only PPE-detection inference service with a measured
  FP32 → INT8 quantization story (OpenVINO), FastAPI API + React demo UI.
- **Owner:** rohityaduvxnshi (yaduvxnshi@gmail.com)
- **Hardware target:** Intel i5-1135G7 (11th-gen, 4C/8T, AVX-512 VNNI present), 16 GB RAM,
  Windows 10 Home 22H2 (build 19045), CPU only, free tools only. Every number in this file
  and the README was measured on this machine unless labelled otherwise.

## Context

The problem: prove an inference-optimization story honestly on commodity CPU hardware —
not train a model. Domain is PPE / construction-safety detection (6 classes). Decisions
and their reasons:

- **Pre-trained YOLOv8n PPE weights** ([Tanishjain9/yolov8n-ppe-detection-6classes](https://huggingface.co/Tanishjain9/yolov8n-ppe-detection-6classes), MIT)
  instead of a YOLO26n fine-tune: the spec's own policy is "pretrained preferred — never
  train a large model on CPU". Usable community weights existed; measured test mAP matches
  the model card, confirming the model↔dataset pairing. A YOLO26n Colab fine-tune notebook
  ([model/train_colab.ipynb](model/train_colab.ipynb)) is provided and drops in without
  pipeline changes.
- **OpenVINO INT8 as the primary CPU path**, calibrated via NNCF from real dataset images.
  ONNX Runtime FP32 kept as the cross-backend baseline; ONNX INT8 measured and *rejected*
  (2× slower — recorded, not hidden).
- **Both served models are OpenVINO** (FP32 + INT8) so the UI toggle demos quantization,
  never a backend swap.
- **React SPA (Vercel) instead of a static page in the container:** richer canvas demo and
  free static hosting; the backend stays a single clean Docker container. The SPA is a
  deliberately thin client of the API contract.
- **HF Spaces (Docker), not a Windows VPS:** the deliverable is a Linux Docker image;
  OpenVINO is best-supported on Linux x86; compute-heavy backends stay off the portfolio VPS.
- **License:** Ultralytics is AGPL-3.0 → this repo is public open source. Weights MIT,
  dataset CC BY 4.0 (Roboflow "PPE Detection_DATA v3").

## Commands

All Python runs through the repo venv (Python 3.12):

```powershell
.venv\Scripts\python -m pytest server\test_api.py -q        # API contract tests (needs data/ppe/test/images)
.venv\Scripts\python -m pytest server\test_api.py -q -k predict   # single test
.venv\Scripts\python -m uvicorn server.app:app --port 8000  # run API locally
.venv\Scripts\python server\loadtest.py -c 1 -n 200         # service-level p50/p95/p99 (server must be running)
.venv\Scripts\python model\export.py                        # re-export ONNX + OpenVINO FP32/INT8
.venv\Scripts\python model\benchmark.py --models model\ppe_yolov8n_openvino_model model\ppe_yolov8n_int8_openvino_model
cd web; npm run dev                                          # frontend (VITE_API_URL in web/.env)
cd web; npm run build                                        # tsc -b && vite build
cd web; npm run lint                                         # oxlint
docker build -f server/Dockerfile -t visionserve .           # CPU-only image (root build context)
deploy\stage_hf_space.ps1                                    # regenerate deploy/hf-space/ staging repo
```

**Benchmark honesty rules (do not violate):**
- Run benchmarks in a **foreground shell on an idle machine** — Windows EcoQoS throttles
  background-launched processes up to 5×, silently corrupting latency numbers. mAP is unaffected.
- Latency comparisons are only valid **same-backend** (OpenVINO FP32 vs OpenVINO INT8).
  The ONNX→OpenVINO delta is a backend win and is reported separately in the README.
- The benchmark interleaves all models round-robin per image on purpose (thermal fairness);
  don't "simplify" it to sequential per-model loops.
- `benchmark.py` merges into `eval/benchmarks.json` per-field, so a `--no-map` rerun keeps stored mAP.
- Never fabricate, estimate, or quote vendor speedups as measurements. Unmeasured = `MEASURE`.

## Architecture

Two halves that meet at the exported OpenVINO model directories:

- **Offline pipeline:** `model/export.py` (PT → ONNX / OpenVINO IR / INT8 with NNCF calibration
  from `data/ppe.yaml`) → `model/benchmark.py` (interleaved latency + test-split mAP →
  `eval/benchmarks.json`) → hand-written analysis in `eval/*.md`.
- **Serving:** `server/app.py` (FastAPI routes, CORS, upload/batch limits, structured JSON
  access log with request id / latency / detection count / status) is a thin shell over
  `server/inference.py`, which owns everything model-related: loads **both** OpenVINO models
  (FP32 + INT8) at lifespan startup, warms them up, and serves predictions under one global
  lock (Ultralytics predictors aren't thread-safe; the lock is also the concurrency cap).
  Inference goes through the Ultralytics `YOLO` wrapper so letterboxing and coordinate
  back-mapping are correct by construction — boxes returned in **original-image pixel space**.
- **Frontend:** `web/` React 19 + Vite SPA, a deliberately thin REST client of the API
  contract in the README (`/predict`, `/predict/batch`, `/health`, `/model/info`). It draws
  canvas boxes and toggles `model=int8|fp32` to demo the latency delta live.
- **Deploy:** `deploy/stage_hf_space.ps1` copies server + model dirs into `deploy/hf-space/`
  (a standalone HF Spaces Docker repo, gitignored); frontend targets Vercel.

Server config is env-driven: `MODEL_DIR`, `DEFAULT_MODEL` (int8), `CONF_THRESHOLD` (0.25),
`ALLOWED_ORIGINS`, `APP_VERSION` (git tag, surfaces in `/health`). Class names are
mixed-case from the dataset (`Gloves`, `Vest`, `goggles`, `helmet`, `mask`, `safety_shoe`)
— tests assert this exact list; don't normalize.

Intentional non-features (decided, don't reopen): no response cache (would fake the latency
demo), no micro-batching, no hard per-request timeout (a sync inference thread can't be
killed safely; work is bounded ~30 ms and the lock caps concurrency — enforce timeouts at
the proxy if ever needed), single-image scope (no video/streams), no `?annotated=true`
server-side rendering (the SPA draws boxes client-side).

## Version history

History note: phases 0–5 were built across sessions and landed collapsed in one commit
(`b87be84`, 2026-07-13) rather than one commit per phase; tags v0.1–v0.4 therefore all
point at that commit, each marking that the phase's acceptance criteria were met there.

- **v0.1–v0.2 — 2026-07-13 — Scaffold, dataset, FP32 baseline** (`b87be84`): repo layout,
  venv, dataset config `data/ppe.yaml` (8774/2070/1234 split), pre-trained weights chosen,
  FP32 baseline mAP measured → [eval/baseline.md](eval/baseline.md).
- **v0.3 — 2026-07-13 — Export + quantize + benchmark** (`b87be84`): `model/export.py`
  (ONNX FP32/INT8, OpenVINO FP32/INT8+calibration), interleaved `model/benchmark.py`,
  results in [eval/quantization.md](eval/quantization.md) + `eval/benchmarks.json`.
- **v0.4 — 2026-07-13 — Inference service** (`b87be84`): FastAPI `/predict`,
  `/predict/batch`, `/health`, `/model/info`, guardrails, 7 contract tests, React UI.
- **v0.5 — 2026-07-13 — Service benchmark + hardening + docs**: `server/loadtest.py`,
  measured service p50/p99 (table below), `/health` enriched to §4 contract
  (model/backend/version), structured JSON access logs, README service section,
  this CLAUDE.md journal.
- **v0.6-rc — 2026-07-13 — Dockerize** (same commit as v0.5; Dockerfile itself landed in
  `b87be84`): CPU-only image, HF Space staging script (`deploy/stage_hf_space.ps1`),
  mojibake encoding bug in staged README fixed. Image size: **2.73 GB** (docker 29.6.1;
  torch-CPU dominates — Ultralytics requires it even for OpenVINO inference). Container
  smoke-tested: `/health` + `/predict` return correctly (2026-07-13).
- **v1.0 — pending — Deploy**: blocked on HF token + portfolio access; see Deployment record.

## Metrics (all measured on the Header hardware unless noted)

In-process inference (interleaved, imgsz=640, batch=1, `OMP_NUM_THREADS=4`, foreground/idle;
full method in [eval/quantization.md](eval/quantization.md)):

| Backend | Precision | p50 (ms) | img/s | mAP@0.5 | mAP@0.5:0.95 | size |
|---|---|---|---|---|---|---|
| PyTorch | FP32 | 102.0 | 9.0 | 0.7905 | 0.4964 | 5.4 MB |
| ONNX RT | FP32 | 75.8 | 13.2 | 0.7869 | 0.4991 | 10.5 MB |
| ONNX RT | INT8 | 145.9 | 6.1 | — | — | 3.0 MB |
| OpenVINO | FP32 | 30.8 | 26.8 | 0.7866 | 0.4996 | 10.6 MB |
| **OpenVINO** | **INT8** | **27.8** | 25.3 | 0.7877 | **0.4946** | **3.2 MB** |

Quantization claim (same backend): **−10% wall p50 sustained / −21% inference-stage only,
−70% model size, −0.005 mAP@0.5:0.95 (−1% relative)** on the 1234-image held-out test split.

Service-level (HTTP end-to-end via `server/loadtest.py`, uvicorn on localhost, 200 warm
requests, single test-split image, 2026-07-13):

| Model | Concurrency | p50 (ms) | p95 (ms) | p99 (ms) | req/s |
|---|---|---|---|---|---|
| INT8 | 1 | 25.9 | 28.4 | 50.4 | 36.3 |
| INT8 | 4 | 90.6 | 104.5 | 145.5 | 38.0 |
| FP32 | 1 | 30.2 | 32.5 | 40.8 | 31.5 |

c=4 latency is queue wait behind the global inference lock (throughput flat ≈37 req/s) —
recorded as the honest concurrent picture, not hidden.

Deployed target (HF Spaces 2 vCPU): `MEASURE` — re-run `model/benchmark.py` and
`server/loadtest.py` on the Space after deploy.

## Deployment record

**Status: NOT DEPLOYED yet.** Blockers checked 2026-07-13 on this machine: no HF token or
`hf` CLI, no `huggingface_hub` in the venv, no Vercel CLI login (Vercel MCP connector *is*
available in Claude Code), no local checkout of the `dash-board.in` portfolio repo.

Planned record (fill each on completion):

- **Backend → HF Docker Space:** create Space `visionserve` (Docker SDK), then
  `deploy\stage_hf_space.ps1` and push `deploy/hf-space/` per the commands in that script's
  header (needs an HF write token). Set `APP_VERSION` to the git tag in the Space settings.
  URL: `MEASURE` · deploy date: `MEASURE` · image tag: `MEASURE`.
- **Frontend → Vercel:** project root `web/`, `VITE_API_URL=<Space URL>`. URL: `MEASURE`.
- **`dash-board.in` registry change** (`projects.config.json`, apply on the portfolio, not
  in this repo): `visionserve` entry → `deployType: "service"` → `"external"`,
  `status: "planned"` → `"live"`, `liveUrl: null` → Space URL, `metrics` → the measured
  table above, drop `servicePort: 8002`. Then redeploy the portfolio.
- **Caddy redirect** on the VPS (redirect, not reverse_proxy — HF Spaces behaves oddly
  behind proxies):
  ```
  visionserve.dash-board.in {
      redir https://<user>-visionserve.hf.space{uri}
  }
  ```
  Redirect status: `MEASURE`.

## Known limitations / honesty log

- **INT8 speedup is real but modest (~10–20%)** on this nano model — memory/overhead bound;
  ~5 ms/request is preprocessing quantization can't touch. The win **requires VNNI**; on
  CPUs without it INT8 can be slower. The HF Space CPU may differ from the dev machine —
  all headline numbers are dev-machine numbers until re-measured there.
- **Honest negative result:** ONNX Runtime INT8 (QDQ static quant) was ~2× *slower* than
  ONNX FP32 on this model despite VNNI.
- **Small-object recall** is the model's real weakness (gloves 38% / shoes 36% / masks 30%
  missed); zero inter-class confusion — see [eval/README.md](eval/README.md).
- **Measurement traps documented:** Windows EcoQoS background throttling (up to 5×);
  thermal state shifts absolute latency ~35% between burst and sustained. Loadtest reuses
  a single image (OS cache favorable) — labelled as such.
- **Concurrency:** one global inference lock; p99 grows linearly with concurrent clients.
  Async queue is the documented next step, deliberately not built at demo scale.
- HF Spaces free tier cold-starts after ~48 h idle. Dataset split integrity relies on
  Roboflow's export. README previously said "Windows 11"; corrected — this machine is
  Windows 10 Home 22H2.
- **Licenses:** Ultralytics AGPL-3.0 (repo must stay open source; no closed commercial
  reuse without an Ultralytics license), weights MIT, dataset CC BY 4.0.

## Version control

- Single `main` branch, remote `origin` = github.com/rohityaduvxnshi/VisionServe (in sync).
  Commit in logical units; tag phases as in Version history.
- Gitignored by design (large or regenerable): `data/*` (except `ppe.yaml`), `*.pt`,
  `*.onnx`, `model/*_openvino_model/`, `runs/`, `web/dist/`, `deploy/hf-space/`, `.venv/`,
  `.env`, `kernel.errors.txt` (Intel graphics-compiler dump). Model artifacts are
  reproducible via `model/export.py`; the dataset (667 MB) re-downloads from Roboflow
  (URL in `data/ppe.yaml` header).

## Project status (maintain this section)

Done: export, benchmarks (in-process + service-level), eval writeups, FastAPI server +
7 contract tests, structured logging, React UI, Dockerfile, HF Space staging script,
loadtest, this journal.

Pending (the `MEASURE` placeholders here and in README.md):
- Deploy backend to HF Spaces (**needs HF write token from owner**) and frontend to Vercel
  (MCP available; do after backend URL exists).
- Apply the dash-board.in registry diff + Caddy redirect (needs portfolio/VPS access).
- Re-run `model/benchmark.py` + `server/loadtest.py` on the Space; fill deployed numbers.
- Record demo GIF; replace `MEASURE` placeholders with live URLs. Tag `v1.0`.
- Optional: YOLO26n fine-tune via `model/train_colab.ipynb`, per-class conf thresholds,
  tiled inference for small objects.
