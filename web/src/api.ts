import type { ModelInfo, ModelKey, PredictResponse } from "./types";

const API_URL: string = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function predict(
  file: File,
  model: ModelKey,
  conf = 0.25,
): Promise<PredictResponse> {
  const form = new FormData();
  form.append("image", file);
  const res = await fetch(`${API_URL}/predict?model=${model}&conf=${conf}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return res.json();
}

export async function modelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API_URL}/model/info`);
  if (!res.ok) throw new Error(`API ${res.status}`);
  return res.json();
}
