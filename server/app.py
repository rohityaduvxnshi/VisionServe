"""VisionServe API — the seam the frontend builds against (see README API contract)."""

import io
import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from server import inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("visionserve")

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_BATCH = 16
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "int8")
DEFAULT_CONF = float(os.environ.get("CONF_THRESHOLD", "0.25"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    inference.load_models()
    yield


app = FastAPI(title="VisionServe", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_image(upload: UploadFile) -> Image.Image:
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image exceeds {MAX_UPLOAD_BYTES // 1024 // 1024}MB limit")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(415, "file is not a decodable image")


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    logger.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error"})


@app.post("/predict")
def predict(
    image: UploadFile = File(...),
    model: Literal["int8", "fp32"] = Query(DEFAULT_MODEL),
    conf: float = Query(DEFAULT_CONF, ge=0.01, le=1.0),
):
    result = inference.predict(read_image(image), model, conf)
    logger.info("predict model=%s conf=%.2f ms=%.1f detections=%d",
                model, conf, result["inference_ms"], len(result["detections"]))
    return result


@app.post("/predict/batch")
def predict_batch(
    images: list[UploadFile] = File(...),
    model: Literal["int8", "fp32"] = Query(DEFAULT_MODEL),
    conf: float = Query(DEFAULT_CONF, ge=0.01, le=1.0),
):
    if len(images) > MAX_BATCH:
        raise HTTPException(413, f"batch limited to {MAX_BATCH} images")
    return {"results": [inference.predict(read_image(f), model, conf) for f in images]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/info")
def model_info():
    return inference.model_info()
