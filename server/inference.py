"""Model loading + prediction. All inference logic lives server-side, here.

Both models are OpenVINO (FP32 IR and INT8) so the API's model toggle shows the
honest same-backend quantization delta, not a backend swap. The Ultralytics
wrapper handles letterbox preprocessing and maps boxes back to original-image
pixel space (r.boxes.xyxy is already in original coordinates).
"""

import os
import threading
import time
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "model"))
IMGSZ = 640
VERSION = "yolov8n-ppe-v1"

_PATHS = {
    "fp32": MODEL_DIR / "ppe_yolov8n_openvino_model",
    "int8": MODEL_DIR / "ppe_yolov8n_int8_openvino_model",
}
_models: dict[str, YOLO] = {}
_lock = threading.Lock()  # ponytail: one global predict lock (ultralytics predictors aren't thread-safe); per-model locks if throughput matters


def load_models() -> None:
    for key, path in _PATHS.items():
        _models[key] = YOLO(str(path), task="detect")
        _models[key].predict(Image.new("RGB", (IMGSZ, IMGSZ)), imgsz=IMGSZ, verbose=False)  # warmup


def model_info() -> dict:
    names = _models["int8"].names
    return {
        "format": "openvino-int8",
        "classes": [names[i] for i in sorted(names)],
        "input_size": IMGSZ,
        "version": VERSION,
    }


def predict(img: Image.Image, model_key: str, conf: float) -> dict:
    model = _models[model_key]
    with _lock:
        t0 = time.perf_counter()
        r = model.predict(img, imgsz=IMGSZ, conf=conf, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
    return {
        "model": model_key,
        "inference_ms": round(ms, 1),
        "image": {"width": img.width, "height": img.height},
        "detections": [
            {
                "class_id": int(c),
                "class_name": model.names[int(c)],
                "confidence": round(float(cf), 4),
                "box": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
            }
            for (x1, y1, x2, y2), cf, c in zip(
                r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()
            )
        ],
    }
