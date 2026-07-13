import { useEffect, useRef, useState } from "react";
import { predict } from "./api";
import { DetectionList } from "./components/DetectionList";
import { Header } from "./components/Header";
import { ImageCanvas } from "./components/ImageCanvas";
import { LatencyBadge } from "./components/LatencyBadge";
import { ModelToggle } from "./components/ModelToggle";
import { UploadDropzone } from "./components/UploadDropzone";
import type { ModelKey, PredictResponse } from "./types";

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [model, setModel] = useState<ModelKey>("int8");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [latencies, setLatencies] = useState<Partial<Record<ModelKey, number>>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestSeq = useRef(0);

  const handleFile = (f: File) => {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    setFile(f);
    setImageUrl(URL.createObjectURL(f));
    setResult(null);
    setLatencies({});
  };

  useEffect(() => {
    if (!file) return;
    const seq = ++requestSeq.current;
    setLoading(true);
    setError(null);
    // median of 3 runs: single-request latency is too noisy for a fair FP32/INT8 comparison
    (async () => {
      const runs = [];
      for (let i = 0; i < 3; i++) runs.push(await predict(file, model));
      return runs;
    })()
      .then((runs) => {
        if (seq !== requestSeq.current) return; // a newer request superseded this one
        const ms = runs.map((r) => r.inference_ms).sort((a, b) => a - b)[1];
        setResult(runs[runs.length - 1]);
        setLatencies((prev) => ({ ...prev, [model]: ms }));
      })
      .catch((e) => seq === requestSeq.current && setError(String(e)))
      .finally(() => seq === requestSeq.current && setLoading(false));
  }, [file, model]);

  return (
    <div className="app">
      <Header />
      <main className="main">
        <section className="viewer">
          <UploadDropzone onFile={handleFile} />
          {imageUrl && (
            <div className={loading ? "canvas-wrap canvas-wrap--loading" : "canvas-wrap"}>
              <ImageCanvas src={imageUrl} detections={result?.detections ?? []} />
            </div>
          )}
        </section>
        <aside className="sidebar">
          <ModelToggle model={model} onChange={setModel} disabled={!file || loading} />
          <LatencyBadge latencies={latencies} active={model} />
          {error && <p className="error">{error}</p>}
          {result && <DetectionList detections={result.detections} />}
        </aside>
      </main>
    </div>
  );
}
