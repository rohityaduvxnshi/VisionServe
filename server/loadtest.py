"""Service-level load test: latency as an HTTP client sees it (decode + queue + inference).

This is the SERVICE number (Phase 4) — distinct from model/benchmark.py, which measures
in-process predict() only. The delta between them is HTTP + multipart + queueing overhead.

Run the server in a separate FOREGROUND shell on an idle machine (EcoQoS — see benchmark.py):
  .venv\\Scripts\\python -m uvicorn server.app:app --port 8000
Then:
  .venv\\Scripts\\python server\\loadtest.py -c 1 -n 200
  .venv\\Scripts\\python server\\loadtest.py -c 4 -n 200   # queueing under concurrency

Note: inference is serialized under a global lock (Ultralytics predictors aren't
thread-safe), so with -c 4 the p99 measures queue wait — that's the honest number
a concurrent client would see, not a defect of the test.
"""

import argparse
import threading
import time
from pathlib import Path

import httpx

WARMUP = 10


def pct(ts: list[float], p: float) -> float:
    return sorted(ts)[min(len(ts) - 1, int(len(ts) * p / 100))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--image", default=None, help="defaults to first test-split image")
    ap.add_argument("--model", default="int8", choices=["int8", "fp32"])
    ap.add_argument("-c", "--concurrency", type=int, default=1)
    ap.add_argument("-n", "--requests", type=int, default=200)
    args = ap.parse_args()

    img = Path(args.image) if args.image else next(Path("data/ppe/test/images").glob("*.jpg"))
    data = img.read_bytes()
    endpoint = f"{args.url}/predict?model={args.model}"

    with httpx.Client(timeout=60) as c:
        for _ in range(WARMUP):
            c.post(endpoint, files={"image": data}).raise_for_status()

    times: list[float] = []
    errors = 0
    lock = threading.Lock()
    remaining = args.requests

    def worker():
        nonlocal remaining, errors
        with httpx.Client(timeout=60) as c:
            while True:
                with lock:
                    if remaining <= 0:
                        return
                    remaining -= 1
                t0 = time.perf_counter()
                r = c.post(endpoint, files={"image": data})
                dt = (time.perf_counter() - t0) * 1000
                with lock:
                    if r.status_code == 200:
                        times.append(dt)
                    else:
                        errors += 1

    t_start = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_s = time.perf_counter() - t_start

    print(f"\nmodel={args.model} concurrency={args.concurrency} n={len(times)} errors={errors} image={img.name}")
    print(f"p50={pct(times, 50):.1f}ms  p95={pct(times, 95):.1f}ms  p99={pct(times, 99):.1f}ms  "
          f"throughput={len(times) / wall_s:.1f} req/s")


if __name__ == "__main__":
    main()
