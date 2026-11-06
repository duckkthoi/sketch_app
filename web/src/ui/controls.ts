import { formatNumber, h } from "../dom";
import type { ParamSpec, ParamValue, Params } from "../types";

/** Sinh form tham số từ schema của server: float/int -> thanh trượt, choice -> select/segmented, bool -> công tắc. */
export function renderControls(
  root: HTMLElement,
  specs: ParamSpec[],
  values: Params,
  onChange: (key: string, value: ParamValue) => void,
): void {
  root.replaceChildren(...specs.map((s) => control(s, values[s.key] ?? s.default, onChange)));
}

function control(s: ParamSpec, value: ParamValue, onChange: (key: string, value: ParamValue) => void): HTMLElement {
  const id = `p-${s.key}`;
  const help = s.help ? h("small", { class: "muted" }, s.help) : null;

  if (s.kind === "bool") {
    const input = h("input", { type: "checkbox", id, role: "switch" });
    input.checked = Boolean(value);
    input.addEventListener("change", () => onChange(s.key, input.checked));
    return h("div", { class: "ctl ctl-bool" }, h("label", { for: id }, s.label), input, help);
  }

  if (s.kind === "choice") {
    // Ít lựa chọn và nhãn ngắn -> nút phân đoạn; ngược lại dùng select.
    if (s.options.length <= 3 && s.options.reduce((n, o) => n + o.label.length, 0) <= 22) {
      const seg = h("div", { class: "seg seg-sm", role: "radiogroup", "aria-label": s.label });
      for (const o of s.options) {
        const b = h("button", { type: "button", "aria-pressed": String(o.value === value) }, o.label);
        b.addEventListener("click", () => {
          for (const x of Array.from(seg.children)) x.setAttribute("aria-pressed", "false");
          b.setAttribute("aria-pressed", "true");
          onChange(s.key, o.value);
        });
        seg.append(b);
      }
      return h("div", { class: "ctl" }, h("span", { class: "ctl-label" }, s.label), seg, help);
    }
    const sel = h("select", { id });
    for (const o of s.options) sel.append(h("option", { value: o.value, selected: o.value === value }, o.label));
    sel.addEventListener("change", () => onChange(s.key, sel.value));
    return h("div", { class: "ctl" }, h("label", { for: id, class: "ctl-label" }, s.label), sel, help);
  }

  // float / int
  const step = s.step ?? (s.kind === "int" ? 1 : 0.01);
  const range = h("input", {
    type: "range",
    id,
    min: s.min ?? 0,
    max: s.max ?? 1,
    step,
    value: Number(value),
  });
  const out = h("output", { for: id }, formatNumber(Number(value), step));
  const fill = () => {
    const min = s.min ?? 0;
    const max = s.max ?? 1;
    range.style.setProperty("--fill", `${((Number(range.value) - min) / (max - min || 1)) * 100}%`);
  };
  fill();
  range.addEventListener("input", () => {
    const v = s.kind === "int" ? Math.round(Number(range.value)) : Number(range.value);
    out.textContent = formatNumber(v, step);
    fill();
    onChange(s.key, v);
  });
  // Bấm đúp vào thanh trượt để trả về giá trị mặc định.
  range.addEventListener("dblclick", () => {
    range.value = String(s.default);
    range.dispatchEvent(new Event("input"));
  });
  return h(
    "div",
    { class: "ctl" },
    h("div", { class: "ctl-row" }, h("label", { for: id, class: "ctl-label" }, s.label), out),
    range,
    help,
  );
}
