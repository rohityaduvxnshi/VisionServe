import { useEffect, useRef } from "react";
import type { Detection } from "../types";

export const CLASS_COLORS: Record<number, string> = {
  0: "#e6a817", // Gloves
  1: "#2ecc71", // Vest
  2: "#3498db", // goggles
  3: "#e74c3c", // helmet
  4: "#9b59b6", // mask
  5: "#1abc9c", // safety_shoe
};

export function ImageCanvas({
  src,
  detections,
}: {
  src: string;
  detections: Detection[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const img = new Image();
    img.onload = () => {
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext("2d")!;
      ctx.drawImage(img, 0, 0);
      const lw = Math.max(2, img.naturalWidth / 320);
      ctx.font = `${Math.max(14, img.naturalWidth / 45)}px system-ui, sans-serif`;
      ctx.textBaseline = "top";
      for (const d of detections) {
        const { x1, y1, x2, y2 } = d.box;
        const color = CLASS_COLORS[d.class_id] ?? "#ffffff";
        ctx.strokeStyle = color;
        ctx.lineWidth = lw;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const label = `${d.class_name} ${(d.confidence * 100).toFixed(0)}%`;
        const tw = ctx.measureText(label).width;
        const th = parseInt(ctx.font, 10) + 6;
        const ly = y1 >= th ? y1 - th : y1;
        ctx.fillStyle = color;
        ctx.fillRect(x1 - lw / 2, ly, tw + 8, th);
        ctx.fillStyle = "#111";
        ctx.fillText(label, x1 + 3, ly + 3);
      }
    };
    img.src = src;
  }, [src, detections]);

  return <canvas ref={canvasRef} className="image-canvas" />;
}
