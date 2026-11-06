/** Vẽ 2 histogram mức xám (64 bin, đã chuẩn hoá 0..1000) chồng nhau trên canvas. */
export function drawHistogram(canvas: HTMLCanvasElement, input: number[] | null, output: number[] | null): void {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || canvas.width;
  const cssH = canvas.clientHeight || canvas.height;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  const style = getComputedStyle(canvas);
  const area = (data: number[] | null, color: string, alpha: number) => {
    if (!data || data.length === 0) return;
    const n = data.length;
    ctx.beginPath();
    ctx.moveTo(0, cssH);
    data.forEach((v, i) => ctx.lineTo((i / (n - 1)) * cssW, cssH - (v / 1000) * (cssH - 2)));
    ctx.lineTo(cssW, cssH);
    ctx.closePath();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = color;
    ctx.fill();
    ctx.globalAlpha = 1;
  };
  area(input, style.getPropertyValue("--hist-in").trim() || "#999", 0.55);
  area(output, style.getPropertyValue("--hist-out").trim() || "#c2410c", 0.6);
}
