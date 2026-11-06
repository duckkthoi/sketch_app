import { h } from "../dom";
import type { StepImage } from "../types";

/** Lưới ảnh có nhãn (dùng cho "Các bước xử lý" và "So sánh thuật toán"). Bấm vào ảnh để phóng to. */
export function renderGallery(root: HTMLElement, items: StepImage[], numbered = false): void {
  root.replaceChildren(
    ...items.map((it, i) =>
      h(
        "figure",
        { class: "tile" },
        h("button", { type: "button", class: "tile-img", onclick: () => openLightbox(it) },
          h("img", { src: it.image, alt: it.label, loading: "lazy" })),
        h("figcaption", {}, numbered ? h("span", { class: "num" }, String(i + 1)) : null, it.label),
      ),
    ),
  );
}

export function renderPlaceholder(root: HTMLElement, text: string): void {
  root.replaceChildren(h("div", { class: "placeholder muted" }, text));
}

export function openLightbox(item: StepImage): void {
  const close = () => {
    box.remove();
    document.removeEventListener("keydown", onKey);
  };
  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape") close();
  };
  const box = h(
    "div",
    { class: "lightbox", role: "dialog", "aria-modal": "true", "aria-label": item.label, onclick: close },
    h("figure", {}, h("img", { src: item.image, alt: item.label }), h("figcaption", {}, item.label)),
  );
  document.addEventListener("keydown", onKey);
  document.body.append(box);
}
