import { api } from "../api";
import { h } from "../dom";
import type { ResultItem } from "../types";
import { toast } from "./toast";

export interface HistoryHandlers {
  onOpen: (item: ResultItem) => void;
}

/** Tab Lịch sử: lưới thumbnail kết quả đã lưu (lấy từ SQLite qua API), phân trang "Tải thêm". */
export class HistoryView {
  private items: ResultItem[] = [];
  private page = 0;
  private total = 0;
  private loaded = false;
  private loading = false;

  constructor(
    private root: HTMLElement,
    private moreBtn: HTMLButtonElement,
    private handlers: HistoryHandlers,
  ) {
    moreBtn.addEventListener("click", () => void this.loadMore());
  }

  async ensureLoaded(): Promise<void> {
    if (!this.loaded) await this.reload();
  }

  async reload(): Promise<void> {
    this.items = [];
    this.page = 0;
    this.total = 0;
    this.loaded = true;
    await this.loadMore();
  }

  prepend(item: ResultItem): void {
    if (!this.loaded) return;
    this.items.unshift(item);
    this.total += 1;
    this.render();
  }

  private async loadMore(): Promise<void> {
    if (this.loading) return;
    this.loading = true;
    this.moreBtn.disabled = true;
    try {
      const res = await api.results(this.page + 1);
      this.page = res.page;
      this.total = res.total;
      const seen = new Set(this.items.map((i) => i.id));
      this.items.push(...res.items.filter((i) => !seen.has(i.id)));
      this.render();
    } catch (e) {
      toast((e as Error).message, "error");
    } finally {
      this.loading = false;
      this.moreBtn.disabled = false;
    }
  }

  private render(): void {
    this.moreBtn.hidden = this.items.length >= this.total;
    if (this.items.length === 0) {
      this.root.replaceChildren(
        h("div", { class: "placeholder muted" }, "Chưa có kết quả nào. Bấm “Lưu lịch sử” sau khi chỉnh xong một bức vẽ."),
      );
      return;
    }
    this.root.replaceChildren(...this.items.map((it) => this.card(it)));
  }

  private card(it: ResultItem): HTMLElement {
    const date = new Date(it.created_at * 1000).toLocaleString("vi-VN", { dateStyle: "short", timeStyle: "short" });
    const del = async () => {
      if (!confirm("Xoá kết quả này khỏi lịch sử?")) return;
      try {
        await api.deleteResult(it.id);
        this.items = this.items.filter((x) => x.id !== it.id);
        this.total -= 1;
        this.render();
        toast("Đã xoá", "ok");
      } catch (e) {
        toast((e as Error).message, "error");
      }
    };
    return h(
      "article",
      { class: "card" },
      h("a", { class: "card-img", href: it.image_url, target: "_blank", rel: "noopener" },
        h("img", { src: it.thumb_url, alt: `${it.effect_name} – ${it.filename ?? ""}`, loading: "lazy" })),
      h(
        "div",
        { class: "card-body" },
        h("strong", {}, it.effect_name),
        h("span", { class: "muted small", title: it.filename ?? "" }, `${it.filename ?? "ảnh"} · ${it.width}×${it.height}`),
        h("span", { class: "muted small" }, date),
      ),
      h(
        "div",
        { class: "card-actions" },
        h("button", {
          type: "button", class: "btn sm", disabled: !it.image_available,
          title: it.image_available ? "Mở lại ảnh gốc với tham số này" : "Ảnh gốc đã hết hạn",
          onclick: () => this.handlers.onOpen(it),
        }, "Mở lại"),
        h("a", { class: "btn sm", href: `${it.image_url}?download=1` }, "Tải PNG"),
        h("button", { type: "button", class: "btn sm danger", onclick: () => void del() }, "Xoá"),
      ),
    );
  }
}
