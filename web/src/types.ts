// Mirrors the documented API contract (README) — the only coupling to the backend.

export type ModelKey = "int8" | "fp32";

export interface Detection {
  class_id: number;
  class_name: string;
  confidence: number;
  box: { x1: number; y1: number; x2: number; y2: number };
}

export interface PredictResponse {
  model: ModelKey;
  inference_ms: number;
  image: { width: number; height: number };
  detections: Detection[];
}

export interface ModelInfo {
  format: string;
  classes: string[];
  input_size: number;
  version: string;
}
