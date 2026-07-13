# CPU-only serving image. Build from the REPO ROOT:
#   docker build -f server/Dockerfile -t visionserve .
FROM python:3.12-slim

# libgl1/libglib2.0-0: required by opencv-python (ultralytics hard dep)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/requirements.txt server/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r server/requirements.txt

COPY server/app.py server/inference.py server/
COPY model/ppe_yolov8n_openvino_model/ model/ppe_yolov8n_openvino_model/
COPY model/ppe_yolov8n_int8_openvino_model/ model/ppe_yolov8n_int8_openvino_model/

# YOLO_CONFIG_DIR: ultralytics needs a writable settings dir (HF Spaces runs non-root)
ENV YOLO_CONFIG_DIR=/tmp \
    OMP_NUM_THREADS=2

# HF Spaces provides $PORT (7860); default for local runs
EXPOSE 7860
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
