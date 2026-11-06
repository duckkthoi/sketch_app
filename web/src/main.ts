import "./styles.css";

import { ApiError, api } from "./api";
import { $, $$, debounce, h, saveBlob, storage } from "./dom";
import type { EffectSpec, EffectsResponse, ImageInfo, ParamValue, Params, ProcessResponse, ResultItem, Sample, Tab } from "./types";
import { CompareView, type CompareMode } from "./ui/compare";
import { renderControls } from "./ui/controls";
import { renderGallery, renderPlaceholder } from "./ui/gallery";
import { drawHistogram } from "./ui/histogram";
import { HistoryView } from "./ui/history";
import { effectIcon } from "./ui/icons";
import { PresetsView } from "./ui/presets";
import { toast } from "./ui/toast";

// =========================================================================== state
interface State {
  meta: EffectsResponse;
  effect: EffectSpec;
  category: string;
  params: Params;
  image: ImageInfo | null;
  result: ProcessResponse | null;
  tab: Tab;
  analysisKind: "filters" | "edges";
}

let S: State;
let inflight: AbortController | null = null;
let requestSeq = 0;
const analysisCache = new Map<string, { label: string; image: string }[]>();

const el = {
  categories: $("#categories"),
  effects: $("#effects"),
  params: $("#params"),
  desc: $("#effect-desc"),
  empty: $("#empty"),
  stage: $("#stage"),
  samples: $("#samples"),
  steps: $("#steps"),
  analysis: $("#analysis"),
  analysisNote: $("#analysis-note"),
  status: $("#status"),
  info: $("#image-info"),
  hist: $<HTMLCanvasElement>("#hist"),
  file: $<HTMLInputElement>("#file-input"),
  save: $<HTMLButtonElement>("#btn-save"),
  download: $<HTMLButtonElement>("#btn-download"),
  format: $<HTMLSelectElement>("#dl-format"),
};

const compare = new CompareView($("#compare"));
const history = new HistoryView($("#history"), $<HTMLButtonElement>("#history-more"), { onOpen: (it) => void openFromHistory(it) });
const presets = new PresetsView($("#presets"), {
  effectId: () => S.effect.id,
  params: () => S.params,
  apply: (p) => {
    S.params = { ...defaultsFor(S.effect.id), ...p.params };
    renderParams();
    void processNow();
  },
});

// =========================================================================== helpers
function defaultsFor(effectId: string, category = S.category): Params {
  return { ...(S.meta.presets[category]?.[effectId] ?? {}) };
}

function setStatus(text: string): void {
  el.status.textContent = text;
}

function updateHistogram(): void {
  drawHistogram(el.hist, S.image?.histogram ?? null, S.result?.histogram ?? null);
}

function setActionsEnabled(on: boolean): void {
  el.save.disabled = !on;
  el.download.disabled = !on;
}

function handleError(e: unknown): void {
  if ((e as Error).name === "AbortError") return;
  if (e instanceof ApiError && (e.status === 404 || e.status === 410) && S.image) {
    toast("Ảnh đã hết hạn trên máy chủ, hãy tải lại ảnh", "error");
    clearImage();
    return;
  }
  toast((e as Error).message || "Có lỗi xảy ra", "error");
}

// =========================================================================== render
function renderCategories(): void {
  el.categories.replaceChildren(
    ...S.meta.categories.map((c) =>
      h("button", {
        type: "button", role: "radio", "aria-checked": String(c.id === S.category),
        onclick: () => selectCategory(c.id),
      }, c.label),
    ),
  );
}

function renderEffects(): void {
  el.effects.replaceChildren(
    ...S.meta.effects.map((e) =>
      h("button", {
        type: "button", role: "option", class: "effect", "aria-selected": String(e.id === S.effect.id),
        onclick: () => selectEffect(e.id), title: e.description,
      }, effectIcon(e.id), h("span", {}, e.name)),
    ),
  );
}

function renderParams(): void {
  el.desc.textContent = S.effect.description;
  renderControls(el.params, S.effect.params, S.params, onParamChange);
}

function renderSamples(samples: Sample[]): void {
  const groups = S.meta.categories.map((c) => {
    const items = samples.filter((s) => s.category === c.id);
    if (!items.length) return null;
    return h("div", { class: "sample-group" }, h("h4", {}, c.label),
      h("div", { class: "sample-row" }, ...items.map((s) =>
        h("button", { type: "button", class: "sample", onclick: () => void loadSample(s), title: s.name },
          h("img", { src: s.thumb_url, alt: s.name, loading: "lazy" }), h("span", {}, s.name)))));
  });
  el.samples.replaceChildren(...groups.filter((g): g is HTMLDivElement => g !== null));
}

function renderSteps(): void {
  if (!S.image) return renderPlaceholder(el.steps, "Tải một ảnh lên để xem từng bước xử lý.");
  if (!S.result?.steps) return renderPlaceholder(el.steps, "Đang xử lý…");
  renderGallery(el.steps, [{ label: "Ảnh gốc", image: S.image.preview }, ...S.result.steps], true);
}

// =========================================================================== actions
function selectCategory(id: string): void {
  if (id === S.category) return;
  S.category = id;
  S.params = defaultsFor(S.effect.id);
  renderCategories();
  renderParams();
  void processNow();
}

function selectEffect(id: string, params?: Params): void {
  const eff = S.meta.effects.find((e) => e.id === id);
  if (!eff) return;
  S.effect = eff;
  S.params = { ...defaultsFor(id), ...(params ?? {}) };
  storage.set("effect", id);
  renderEffects();
  renderParams();
  presets.render();
  void processNow();
}

const scheduleProcess = debounce(() => void processNow(), 200);

function onParamChange(key: string, value: ParamValue): void {
  S.params[key] = value;
  scheduleProcess();
}

async function processNow(): Promise<void> {
  if (!S.image) return;
  inflight?.abort();
  const ctrl = new AbortController();
  inflight = ctrl;
  const seq = ++requestSeq;
  compare.setBusy(true);
  setStatus("Đang xử lý…");
  try {
    const res = await api.process({ image_id: S.image.id, effect: S.effect.id, params: S.params, steps: true }, ctrl.signal);
    if (seq !== requestSeq) return;
    S.result = res;
    compare.setResult(res.image);
    setActionsEnabled(true);
    renderSteps();
    updateHistogram();
    setStatus(`${S.effect.name} · xử lý ${res.duration_ms} ms trên ảnh preview`);
  } catch (e) {
    if (seq === requestSeq) {
      handleError(e);
      if ((e as Error).name !== "AbortError") setStatus("Xử lý thất bại");
    }
  } finally {
    if (seq === requestSeq) compare.setBusy(false);
  }
}

function setImage(img: ImageInfo): void {
  S.image = img;
  S.result = null;
  analysisCache.clear();
  storage.set("image", img.id);
  if (img.category !== S.category) {
    S.category = img.category;
    S.params = defaultsFor(S.effect.id);
    renderCategories();
    renderParams();
  }
  el.empty.hidden = true;
  el.stage.hidden = false;
  compare.setOriginal(img.preview);
  el.info.textContent = `${img.filename} · ${img.width}×${img.height}px`;
  setActionsEnabled(false);
  updateHistogram();
  renderSteps();
  if (S.tab === "analysis") void loadAnalysis();
}

function clearImage(): void {
  S.image = null;
  S.result = null;
  storage.set("image", "");
  el.empty.hidden = false;
  el.stage.hidden = true;
  setActionsEnabled(false);
  updateHistogram();
  renderSteps();
  renderPlaceholder(el.analysis, "Tải một ảnh lên để so sánh các thuật toán.");
  setStatus("Chưa có ảnh");
}

async function uploadFile(file: File): Promise<void> {
  const max = S.meta.limits.max_upload_mb;
  if (file.size > max * 1024 * 1024) return toast(`File vượt quá ${max} MB`, "error");
  setStatus(`Đang tải lên ${file.name}…`);
  compare.setBusy(true);
  try {
    setImage(await api.upload(file, S.category));
    await processNow();
  } catch (e) {
    handleError(e);
    setStatus("Tải lên thất bại");
  } finally {
    compare.setBusy(false);
  }
}

async function loadSample(s: Sample): Promise<void> {
  setStatus(`Đang mở ảnh mẫu ${s.name}…`);
  try {
    setImage(await api.useSample(s.id));
    await processNow();
  } catch (e) {
    handleError(e);
  }
}

async function openFromHistory(it: ResultItem): Promise<void> {
  if (!it.image_id || !it.image_available) return toast("Ảnh gốc đã hết hạn, chỉ tải được kết quả", "error");
  try {
    const img = await api.image(it.image_id);
    setImage(img);
    switchTab("compare");
    selectEffect(it.effect, it.params);
  } catch (e) {
    handleError(e);
  }
}

async function download(): Promise<void> {
  if (!S.image) return;
  const label = el.download.innerHTML;
  el.download.disabled = true;
  el.download.textContent = "Đang xuất…";
  try {
    const { blob, filename } = await api.download({ image_id: S.image.id, effect: S.effect.id, params: S.params, format: el.format.value });
    saveBlob(blob, filename);
  } catch (e) {
    handleError(e);
  } finally {
    el.download.innerHTML = label;
    el.download.disabled = !S.image;
  }
}

async function saveToHistory(): Promise<void> {
  if (!S.image) return;
  el.save.disabled = true;
  try {
    const r = await api.saveResult({ image_id: S.image.id, effect: S.effect.id, params: S.params });
    history.prepend(r);
    toast("Đã lưu vào lịch sử", "ok");
  } catch (e) {
    handleError(e);
  } finally {
    el.save.disabled = !S.image;
  }
}

async function loadAnalysis(): Promise<void> {
  if (!S.image) return renderPlaceholder(el.analysis, "Tải một ảnh lên để so sánh các thuật toán.");
  const kind = S.analysisKind;
  el.analysisNote.textContent = kind === "filters"
    ? "Cùng độ mịn σ = 2: bộ lọc bảo toàn biên (Bilateral, Kuwahara, Perona-Malik) giữ cạnh sắc nét hơn Gaussian."
    : "Cùng ảnh đã làm mịn Gaussian: gradient bậc 1 (Roberts/Prewitt/Sobel), bậc 2 (Laplacian, LoG) và Canny.";
  const key = `${S.image.id}:${kind}`;
  const cached = analysisCache.get(key);
  if (cached) return renderGallery(el.analysis, cached);
  renderPlaceholder(el.analysis, "Đang tính toán…");
  try {
    const res = await api.compare(S.image.id, kind);
    analysisCache.set(key, res.items);
    if (S.analysisKind === kind) renderGallery(el.analysis, res.items);
  } catch (e) {
    handleError(e);
  }
}

function switchTab(tab: Tab): void {
  S.tab = tab;
  for (const b of $$<HTMLButtonElement>(".tabs [role=tab]")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of ["compare", "steps", "analysis", "history"] as const) $(`#view-${v}`).hidden = v !== tab;
  if (tab === "analysis") void loadAnalysis();
  if (tab === "history") void history.ensureLoaded();
}

// =========================================================================== theme
function applyTheme(theme: string | null): void {
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
}

function currentTheme(): "light" | "dark" {
  const t = document.documentElement.dataset.theme;
  if (t === "light" || t === "dark") return t;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

// =========================================================================== wiring
function wire(): void {
  $("#btn-upload").addEventListener("click", () => el.file.click());
  el.file.addEventListener("change", () => {
    const f = el.file.files?.[0];
    if (f) void uploadFile(f);
    el.file.value = "";
  });
  el.download.addEventListener("click", () => void download());
  el.save.addEventListener("click", () => void saveToHistory());
  $("#btn-reset").addEventListener("click", () => {
    S.params = defaultsFor(S.effect.id);
    renderParams();
    void processNow();
  });
  $("#btn-theme").addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    applyTheme(next);
    storage.set("theme", next);
    updateHistogram();
  });
  for (const b of $$<HTMLButtonElement>(".tabs [role=tab]")) b.addEventListener("click", () => switchTab(b.dataset.tab as Tab));
  for (const b of $$<HTMLButtonElement>("#view-mode button")) {
    b.addEventListener("click", () => {
      for (const x of $$("#view-mode button")) x.setAttribute("aria-pressed", String(x === b));
      compare.setMode(b.dataset.mode as CompareMode);
    });
  }
  for (const b of $$<HTMLButtonElement>("#analysis-kind button")) {
    b.addEventListener("click", () => {
      for (const x of $$("#analysis-kind button")) x.setAttribute("aria-pressed", String(x === b));
      S.analysisKind = b.dataset.kind as "filters" | "edges";
      void loadAnalysis();
    });
  }

  // Kéo thả ảnh vào bất kỳ đâu trên trang.
  let dragDepth = 0;
  window.addEventListener("dragenter", (e) => {
    if (e.dataTransfer?.types.includes("Files")) {
      dragDepth++;
      document.body.classList.add("dragging");
    }
  });
  window.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) document.body.classList.remove("dragging");
  });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    document.body.classList.remove("dragging");
    const f = e.dataTransfer?.files?.[0];
    if (f) void uploadFile(f);
  });
  // Dán ảnh từ clipboard (Ctrl+V).
  window.addEventListener("paste", (e) => {
    const item = Array.from(e.clipboardData?.items ?? []).find((i) => i.type.startsWith("image/"));
    const f = item?.getAsFile();
    if (f) void uploadFile(new File([f], f.name || "clipboard.png", { type: f.type }));
  });
  window.addEventListener("resize", debounce(updateHistogram, 150));
}

async function init(): Promise<void> {
  applyTheme(storage.get("theme"));
  let meta: EffectsResponse;
  try {
    // Gọi /api/effects trước (một mình) để server cấp cookie client_id trước các request khác.
    meta = await api.effects();
  } catch (e) {
    setStatus("Không kết nối được máy chủ");
    toast((e as Error).message, "error", 8000);
    return;
  }
  const savedEffect = storage.get("effect");
  const effect = meta.effects.find((e) => e.id === savedEffect) ?? meta.effects.find((e) => e.id === "sketch") ?? meta.effects[0];
  S = { meta, effect, category: "nature", params: {}, image: null, result: null, tab: "compare", analysisKind: "filters" };
  S.params = defaultsFor(effect.id);
  $("#max-mb").textContent = String(meta.limits.max_upload_mb);

  renderCategories();
  renderEffects();
  renderParams();
  renderSteps();
  renderPlaceholder(el.analysis, "Tải một ảnh lên để so sánh các thuật toán.");
  wire();
  updateHistogram();

  const [samples] = await Promise.all([api.samples().catch(() => [] as Sample[]), presets.load()]);
  renderSamples(samples);

  // Mở lại ảnh của lần trước (nếu còn trên server).
  const last = storage.get("image");
  if (last) {
    try {
      setImage(await api.image(last));
      await processNow();
    } catch {
      storage.set("image", "");
    }
  }
}

void init();
