import { h } from "../dom";

export type CompareMode = "split" | "result" | "original";

/** Khung so sánh trước/sau: kéo thanh dọc để xem ảnh gốc (trái) và kết quả (phải). */
export class CompareView {
  private before = h("img", { class: "cmp-img", alt: "Ảnh gốc", draggable: "false" });
  private after = h("img", { class: "cmp-img cmp-after", alt: "Kết quả", draggable: "false" });
  private handle = h(
    "div",
    { class: "cmp-handle", role: "slider", tabindex: 0, "aria-label": "Vị trí so sánh", "aria-valuemin": 0, "aria-valuemax": 100 },
    h("span", { class: "cmp-knob" }),
  );
  private spinner = h("div", { class: "spinner", hidden: true }, h("span"));
  private frame = h("div", { class: "cmp-frame" }, this.before, this.after, this.handle, this.spinner);
  private pos = 50;
  private mode: CompareMode = "split";

  constructor(root: HTMLElement) {
    root.replaceChildren(
      this.frame,
      h("span", { class: "cmp-tag left" }, "Ảnh gốc"),
      h("span", { class: "cmp-tag right" }, "Kết quả"),
    );
    this.setPos(50);
    const move = (clientX: number) => {
      const r = this.frame.getBoundingClientRect();
      this.setPos(((clientX - r.left) / r.width) * 100);
    };
    this.frame.addEventListener("pointerdown", (e) => {
      if (this.mode !== "split") return;
      this.frame.setPointerCapture(e.pointerId);
      move(e.clientX);
      const onMove = (ev: PointerEvent) => move(ev.clientX);
      const onUp = () => {
        this.frame.removeEventListener("pointermove", onMove);
        this.frame.removeEventListener("pointerup", onUp);
      };
      this.frame.addEventListener("pointermove", onMove);
      this.frame.addEventListener("pointerup", onUp);
    });
    this.handle.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft") this.setPos(this.pos - 5);
      else if (e.key === "ArrowRight") this.setPos(this.pos + 5);
      else return;
      e.preventDefault();
    });
  }

  setOriginal(src: string): void {
    this.before.src = src;
    this.after.removeAttribute("src");
  }

  setResult(src: string): void {
    this.after.src = src;
  }

  setBusy(busy: boolean): void {
    this.spinner.hidden = !busy;
    this.frame.classList.toggle("busy", busy);
  }

  setMode(mode: CompareMode): void {
    this.mode = mode;
    this.frame.dataset.mode = mode;
    this.applyClip();
  }

  private setPos(p: number): void {
    this.pos = Math.max(0, Math.min(100, p));
    this.handle.setAttribute("aria-valuenow", String(Math.round(this.pos)));
    this.applyClip();
  }

  private applyClip(): void {
    const p = this.mode === "split" ? this.pos : this.mode === "result" ? 0 : 100;
    this.after.style.clipPath = `inset(0 0 0 ${p}%)`;
    this.handle.style.left = `${p}%`;
    this.handle.hidden = this.mode !== "split";
  }
}
