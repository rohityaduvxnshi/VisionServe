import type { Detection } from "../types";
import { CLASS_COLORS } from "./ImageCanvas";

export function DetectionList({ detections }: { detections: Detection[] }) {
  if (detections.length === 0) return <p className="muted">No detections.</p>;
  return (
    <ul className="detection-list">
      {detections.map((d, i) => (
        <li key={i}>
          <span
            className="color-chip"
            style={{ background: CLASS_COLORS[d.class_id] ?? "#fff" }}
          />
          <span>{d.class_name}</span>
          <span className="muted">{(d.confidence * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ul>
  );
}
