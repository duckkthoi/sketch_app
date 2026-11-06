import { api } from "../api";
import { h } from "../dom";
import type { Params, Preset } from "../types";
import { toast } from "./toast";

export interface PresetHost {
  effectId: () => string;
  params: () => Params;
  apply: (p: Preset) => void;
}

/** Preset người dùng tự đặt tên, lưu trong SQLite theo trình duyệt. */
export class PresetsView {
  private items: Preset[] = [];

  constructor(
    private root: HTMLElement,
    private host: PresetHost,
  ) {}

  async load(): Promise<void> {
    try {
      this.items = await api.presets();
    } catch {
      this.items = [];
    }
    this.render();
  }

  render(): void {
    const mine = this.items.filter((p) => p.effect === this.host.effectId());
    const sel = h("select", { "aria-label": "Chọn preset" }, h("option", { value: "" }, mine.length ? "— Chọn preset —" : "Chưa có preset"));
    for (const p of mine) sel.append(h("option", { value: String(p.id) }, p.name));
    sel.disabled = mine.length === 0;
    sel.addEventListener("change", () => {
      const p = mine.find((x) => String(x.id) === sel.value);
      if (p) this.host.apply(p);
      delBtn.disabled = !p;
    });

    const delBtn = h("button", { type: "button", class: "btn sm danger", disabled: true, title: "Xoá preset đang chọn" }, "Xoá");
    delBtn.addEventListener("click", async () => {
      const id = Number(sel.value);
      if (!id) return;
      try {
        await api.deletePreset(id);
        this.items = this.items.filter((p) => p.id !== id);
        this.render();
        toast("Đã xoá preset", "ok");
      } catch (e) {
        toast((e as Error).message, "error");
      }
    });

    const name = h("input", { type: "text", placeholder: "Tên preset…", maxlength: 60, "aria-label": "Tên preset" });
    const form = h("form", { class: "preset-form" }, name, h("button", { type: "submit", class: "btn sm" }, "Lưu"));
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      if (!name.value.trim()) {
        name.focus();
        return;
      }
      try {
        const p = await api.createPreset(name.value, this.host.effectId(), this.host.params());
        this.items.unshift(p);
        this.render();
        toast(`Đã lưu preset “${p.name}”`, "ok");
      } catch (err) {
        toast((err as Error).message, "error");
      }
    });

    this.root.replaceChildren(h("div", { class: "preset-row" }, sel, delBtn), form);
  }
}
