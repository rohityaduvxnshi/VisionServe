"""Export the PPE detector: ONNX FP32, OpenVINO FP32 IR, OpenVINO INT8 (calibrated).

INT8 calibration uses images from data/ppe.yaml via NNCF (ultralytics handles it).
Note: ultralytics >= 8.4 uses quantize=8 (int8=True is deprecated).
Run AFTER the FP32 baseline is recorded in eval/baseline.md.
"""

from ultralytics import YOLO

# ponytail: paths hardcoded to this repo's one model; parameterize if a second model lands
PT = "model/ppe_yolov8n.pt"
DATA = "data/ppe.yaml"
IMGSZ = 640

model = YOLO(PT)
# INT8 ONNX first: its export pipeline writes the FP32 .onnx as an intermediate
# and deletes it, so exporting it before FP32 avoids clobbering the FP32 artifact
print("-> ONNX INT8 (ORT static quant, calibrated):", model.export(format="onnx", imgsz=IMGSZ, quantize=8, data=DATA))
print("-> ONNX FP32:", model.export(format="onnx", imgsz=IMGSZ, dynamic=False))
print("-> OpenVINO FP32 IR:", model.export(format="openvino", imgsz=IMGSZ))
print("-> OpenVINO INT8 (calibrated):", model.export(format="openvino", imgsz=IMGSZ, quantize=8, data=DATA))
