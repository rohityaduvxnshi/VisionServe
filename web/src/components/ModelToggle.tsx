import type { ModelKey } from "../types";

export function ModelToggle({
  model,
  onChange,
  disabled,
}: {
  model: ModelKey;
  onChange: (m: ModelKey) => void;
  disabled?: boolean;
}) {
  return (
    <div className="model-toggle" role="group" aria-label="Model precision">
      {(["int8", "fp32"] as const).map((m) => (
        <button
          key={m}
          className={m === model ? "toggle-btn toggle-btn--active" : "toggle-btn"}
          onClick={() => onChange(m)}
          disabled={disabled}
        >
          {m.toUpperCase()}
        </button>
      ))}
    </div>
  );
}
