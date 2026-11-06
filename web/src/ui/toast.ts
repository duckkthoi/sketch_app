import { $, h } from "../dom";

export function toast(message: string, kind: "info" | "error" | "ok" = "info", ms = 3500): void {
  const box = $("#toasts");
  const el = h("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status" }, message);
  box.append(el);
  requestAnimationFrame(() => el.classList.add("show"));
  setTimeout(() => {
    el.classList.remove("show");
    setTimeout(() => el.remove(), 250);
  }, ms);
}
