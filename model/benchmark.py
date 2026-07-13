"""Latency + mAP benchmark for VisionServe model artifacts.

Honesty rules baked in:
- every artifact runs on this same machine through the same Ultralytics wrapper,
  so same-backend comparisons (e.g. OpenVINO FP32 vs INT8) are apples-to-apples
- thread config is fixed where the backend honors it and recorded in the output
- latency is end-to-end predict() (preprocess + inference + postprocess) over
  real dataset images, warm runs only; mAP is computed on the held-out test split
- all models are benchmarked INTERLEAVED (round-robin per image): on a
  thermally-limited laptop CPU, absolute latency drifts as the chip heats, so
  sequential per-model blocks are unfair. Interleaving gives every model the
  same instantaneous conditions, which is what makes the FP32-vs-INT8 delta
  trustworthy even when absolute numbers wobble.

Usage:
  python model/benchmark.py --models model/ppe_yolov8n.pt --data data/ppe.yaml
  python model/benchmark.py --models m1_openvino_model m1_int8_openvino_model --runs 200

WARNING: run latency benchmarks in a FOREGROUND shell on an idle machine.
Windows 11 throttles background-launched processes (EcoQoS), which inflated
p50 up to 5x in testing. mAP is unaffected by scheduling.
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path

N_THREADS = int(os.environ.get("OMP_NUM_THREADS") or "4")
os.environ["OMP_NUM_THREADS"] = str(N_THREADS)  # must be set before torch import

import numpy as np
import torch
import yaml
from ultralytics import YOLO

torch.set_num_threads(N_THREADS)


def bench_images(data_yaml: Path, limit: int = 50) -> list[Path]:
    cfg = yaml.safe_load(data_yaml.read_text())
    root = (data_yaml.parent / cfg.get("path", ".")).resolve()
    val_dir = root / cfg["val"]
    imgs = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    assert imgs, f"no images in {val_dir}"
    return imgs[:limit]


def bench_latency(model_paths: list[str], images: list[Path], runs: int, imgsz: int) -> dict[str, dict]:
    models = {p: YOLO(p, task="detect") for p in model_paths}
    for m in models.values():
        for _ in range(20):  # warmup
            m.predict(images[0], imgsz=imgsz, verbose=False)
    times = {p: [] for p in model_paths}
    for i in range(runs):  # round-robin: same image, same moment, every model
        img = images[i % len(images)]
        for p, m in models.items():
            t0 = time.perf_counter()
            m.predict(img, imgsz=imgsz, verbose=False)
            times[p].append((time.perf_counter() - t0) * 1000)
    out = {}
    for p, ts in times.items():
        p50, p95, p99 = np.percentile(ts, [50, 95, 99])
        out[p] = {
            "model": p,
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
            "p99_ms": round(p99, 1),
            "img_per_s": round(1000 / np.mean(ts), 2),
            "runs": runs,
        }
    return out


def add_map(result: dict, data: str, imgsz: int) -> None:
    metrics = YOLO(result["model"], task="detect").val(
        data=data, split="test", imgsz=imgsz, verbose=False, plots=False)
    result["map50"] = round(float(metrics.box.map50), 4)
    result["map50_95"] = round(float(metrics.box.map), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", default="data/ppe.yaml")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--out", default="eval/benchmarks.json")
    args = ap.parse_args()

    images = bench_images(Path(args.data))
    results = {
        "cpu": platform.processor(),
        "cores": os.cpu_count(),
        "omp_threads": N_THREADS,
        "torch_threads": torch.get_num_threads(),
        "note": "OpenVINO (TBB) and ONNX Runtime manage their own thread pools; "
                "both default to physical cores. Same machine + same wrapper for every row.",
        "imgsz": args.imgsz,
        "results": [],
    }
    print(f"interleaved latency benchmark: {len(args.models)} models x {args.runs} runs ...")
    latency = bench_latency(args.models, images, args.runs, args.imgsz)
    for m in args.models:
        if not args.no_map:
            print(f"mAP eval: {m} ...")
            add_map(latency[m], args.data, args.imgsz)
        results["results"].append(latency[m])

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    existing = json.loads(out.read_text())["results"] if out.exists() else []
    merged = {r["model"]: r for r in existing}
    for r in results["results"]:  # per-field merge so a --no-map rerun keeps stored mAP
        merged[r["model"]] = merged.get(r["model"], {}) | r
    results["results"] = list(merged.values())
    out.write_text(json.dumps(results, indent=2))

    print(f"\n{results['cpu']} | {results['cores']} logical cores | OMP/torch threads: {N_THREADS}")
    print("| Model | p50 (ms) | p95 (ms) | p99 (ms) | img/s | mAP@0.5 | mAP@0.5:0.95 |")
    print("|---|---|---|---|---|---|---|")
    for r in results["results"]:
        print(f"| {r['model']} | {r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} "
              f"| {r['img_per_s']} | {r.get('map50', '-')} | {r.get('map50_95', '-')} |")


if __name__ == "__main__":
    main()
