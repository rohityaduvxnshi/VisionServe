import type { ModelKey } from "../types";

export function LatencyBadge({
  latencies,
  active,
}: {
  latencies: Partial<Record<ModelKey, number>>;
  active: ModelKey;
}) {
  const ms = latencies[active];
  if (ms === undefined) return null;
  const other: ModelKey = active === "int8" ? "fp32" : "int8";
  const otherMs = latencies[other];
  return (
    <div className="latency-badge">
      <strong>
        {active.toUpperCase()}: {ms.toFixed(1)} ms
      </strong>
      {otherMs !== undefined && (
        <span className="muted">
          {" "}
          vs {other.toUpperCase()} {otherMs.toFixed(1)} ms (
          {(((ms - otherMs) / otherMs) * 100).toFixed(0)}%)
        </span>
      )}
    </div>
  );
}
