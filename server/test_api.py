"""API contract checks (Phase 3 acceptance). Run from repo root: pytest server/test_api.py"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.app import app

TEST_IMG = next(Path("data/ppe/test/images").glob("*.jpg"))


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # context manager triggers lifespan (model load)
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["backend"] == "openvino"
    assert body["model"] and body["version"]


def test_model_info(client):
    info = client.get("/model/info").json()
    assert info["classes"] == ["Gloves", "Vest", "goggles", "helmet", "mask", "safety_shoe"]
    assert info["input_size"] == 640


def test_predict_both_models(client):
    for model in ("int8", "fp32"):
        r = client.post(f"/predict?model={model}", files={"image": TEST_IMG.read_bytes()})
        assert r.status_code == 200
        body = r.json()
        assert body["model"] == model
        assert 1 < body["inference_ms"] < 5000
        assert body["image"]["width"] > 0
        for d in body["detections"]:
            b = d["box"]
            assert 0 <= b["x1"] < b["x2"] <= body["image"]["width"]
            assert 0 <= b["y1"] < b["y2"] <= body["image"]["height"]
            assert 0 < d["confidence"] <= 1


def test_batch(client):
    r = client.post("/predict/batch", files=[("images", TEST_IMG.read_bytes())] * 2)
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_rejects_non_image(client):
    assert client.post("/predict", files={"image": b"not an image"}).status_code == 415


def test_rejects_oversized(client):
    big = io.BytesIO(b"\xff" * (11 * 1024 * 1024))
    assert client.post("/predict", files={"image": big.read()}).status_code == 413


def test_rejects_bad_model_param(client):
    r = client.post("/predict?model=fp16", files={"image": TEST_IMG.read_bytes()})
    assert r.status_code == 422
