import type {
  CompareResponse,
  EffectsResponse,
  ImageInfo,
  Params,
  Preset,
  ProcessResponse,
  ResultItem,
  ResultsPage,
  Sample,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

type Body = FormData | Record<string, unknown> | undefined;

async function send(method: string, url: string, body?: Body, signal?: AbortSignal): Promise<Response> {
  const init: RequestInit = { method, signal, credentials: "same-origin", headers: {} };
  if (body instanceof FormData) {
    init.body = body;
  } else if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  let r: Response;
  try {
    r = await fetch(url, init);
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError(0, "Không kết nối được máy chủ");
  }
  if (!r.ok) {
    let msg = `Lỗi ${r.status}`;
    try {
      const js = (await r.json()) as { error?: string };
      if (js.error) msg = js.error;
    } catch {
      /* phản hồi không phải JSON */
    }
    throw new ApiError(r.status, msg);
  }
  return r;
}

async function json<T>(method: string, url: string, body?: Body, signal?: AbortSignal): Promise<T> {
  return (await send(method, url, body, signal)).json() as Promise<T>;
}

function filenameFrom(r: Response, fallback: string): string {
  const cd = r.headers.get("Content-Disposition") ?? "";
  const star = /filename\*=UTF-8''([^;]+)/i.exec(cd);
  if (star) return decodeURIComponent(star[1]);
  const plain = /filename="?([^";]+)"?/i.exec(cd);
  return plain ? plain[1] : fallback;
}

export interface RenderRequest {
  image_id: string;
  effect: string;
  params: Params;
}

export const api = {
  effects: () => json<EffectsResponse>("GET", "/api/effects"),
  samples: () => json<{ samples: Sample[] }>("GET", "/api/samples").then((r) => r.samples),
  useSample: (id: string) => json<{ image: ImageInfo }>("POST", "/api/samples/use", { id }).then((r) => r.image),
  upload(file: File, category: string): Promise<ImageInfo> {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("category", category);
    return json<{ image: ImageInfo }>("POST", "/api/upload", fd).then((r) => r.image);
  },
  image: (id: string) => json<{ image: ImageInfo }>("GET", `/api/images/${id}`).then((r) => r.image),
  process: (req: RenderRequest & { steps: boolean }, signal?: AbortSignal) =>
    json<ProcessResponse>("POST", "/api/process", { ...req }, signal),
  compare: (image_id: string, kind: "filters" | "edges", signal?: AbortSignal) =>
    json<CompareResponse>("POST", "/api/compare", { image_id, kind }, signal),
  async download(req: RenderRequest & { format: string }): Promise<{ blob: Blob; filename: string }> {
    const r = await send("POST", "/api/download", { ...req });
    return { blob: await r.blob(), filename: filenameFrom(r, `ket-qua.${req.format}`) };
  },
  saveResult: (req: RenderRequest) => json<{ result: ResultItem }>("POST", "/api/results", { ...req }).then((r) => r.result),
  results: (page: number) => json<ResultsPage>("GET", `/api/results?page=${page}&per_page=24`),
  deleteResult: (id: string) => json<{ ok: boolean }>("DELETE", `/api/results/${id}`),
  presets: () => json<{ items: Preset[] }>("GET", "/api/presets").then((r) => r.items),
  createPreset: (name: string, effect: string, params: Params) =>
    json<{ preset: Preset }>("POST", "/api/presets", { name, effect, params }).then((r) => r.preset),
  deletePreset: (id: number) => json<{ ok: boolean }>("DELETE", `/api/presets/${id}`),
};
