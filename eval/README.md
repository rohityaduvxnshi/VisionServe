# Evaluation report — held-out test split (1234 images, 1782 instances), 2026-07-02

Both served models evaluated with `model.val(data="data/ppe.yaml", split="test", imgsz=640)`,
ultralytics 8.4.84, CPU. Plots (confusion matrices, PR/P/R/F1 curves) are in
[`fp32_openvino/`](fp32_openvino/) and [`int8_openvino/`](int8_openvino/).
The PyTorch-FP32 baseline is in [`baseline.md`](baseline.md); latency comparison in
[`quantization.md`](quantization.md); raw numbers in [`benchmarks.json`](benchmarks.json).

## Overall

| Metric | OpenVINO FP32 | OpenVINO INT8 | delta |
|---|---|---|---|
| mAP@0.5 | 0.7866 | 0.7877 | +0.0011 |
| mAP@0.5:0.95 | 0.4996 | 0.4946 | **−0.0050 (−1.0% rel.)** |
| Precision (all) | 0.845 | 0.832 | −0.013 |
| Recall (all) | 0.711 | 0.726 | +0.015 |

INT8 quantization cost 1% relative mAP@0.5:0.95 — within the expected <1–2% for
calibrated post-training quantization. At the looser IoU threshold (mAP@0.5) INT8 is
indistinguishable from FP32; the loss is in localization tightness, which is exactly
where reduced numeric precision should show up.

## Per-class (P / R / mAP@0.5 / mAP@0.5:0.95)

| Class | FP32 | INT8 |
|---|---|---|
| Gloves | 0.763 / 0.540 / 0.627 / 0.453 | 0.750 / 0.543 / 0.623 / 0.446 |
| Vest | 0.861 / 0.922 / 0.932 / 0.736 | 0.852 / 0.938 / 0.930 / 0.726 |
| goggles | 0.902 / 0.866 / 0.923 / 0.491 | 0.909 / 0.878 / 0.936 / 0.491 |
| helmet | 0.882 / 0.764 / 0.880 / 0.580 | 0.867 / 0.791 / 0.882 / 0.577 |
| mask | 0.892 / 0.640 / 0.719 / 0.391 | 0.873 / 0.651 / 0.734 / 0.390 |
| safety_shoe | 0.768 / 0.537 / 0.639 / 0.347 | 0.743 / 0.553 / 0.621 / 0.337 |

No class degrades disproportionately under INT8 (worst: safety_shoe −0.010 mAP@0.5:0.95),
i.e. calibration didn't sacrifice a minority class.

## Confusion matrix read (INT8; FP32 is qualitatively identical)

The striking feature is that **inter-class confusion is essentially zero** — there is no
off-diagonal mass between PPE classes. The model never calls a vest a helmet. All error
mass sits in the background row/column:

- **Misses (true class → predicted background):** Gloves 38%, safety_shoe 36%, mask 30%,
  helmet 15%, goggles 6%, Vest 4%. The weak classes are the physically small objects —
  gloves, shoes and masks occupy tiny pixel areas in wide construction-site shots.
- **False positives (background → predicted class):** helmet 26%, Vest 25%, Gloves 25%
  of background cells — helmet-like round objects and hi-vis-colored regions get picked up.

Practical implication: to improve this system you would not touch the classifier head —
you would improve small-object recall (higher input resolution, tiling, or a larger
backbone), and the per-class confidence thresholds for gloves/shoes could be lowered at
the cost of precision.

## Caveats

- Test split ships with the dataset (Roboflow); leakage between augmented variants across
  splits can't be fully audited. Metrics vs the model card's self-reported val numbers
  (~0.81 mAP@0.5) are consistent, which supports split integrity.
- 373 of 1234 test images contain no labeled objects (negatives); they contribute to the
  background false-positive rates above.
- Single mixed segment annotation in the test labels (dropped by ultralytics, boxes kept).
