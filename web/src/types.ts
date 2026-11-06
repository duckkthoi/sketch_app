export type ParamKind = "float" | "int" | "choice" | "bool";
export type ParamValue = number | string | boolean;
export type Params = Record<string, ParamValue>;

export interface ParamOption {
  value: string;
  label: string;
}

export interface ParamSpec {
  key: string;
  label: string;
  kind: ParamKind;
  default: ParamValue;
  min: number | null;
  max: number | null;
  step: number | null;
  options: ParamOption[];
  help: string;
}

export interface EffectSpec {
  id: string;
  name: string;
  description: string;
  params: ParamSpec[];
}

export interface Category {
  id: string;
  label: string;
}

export interface EffectsResponse {
  effects: EffectSpec[];
  categories: Category[];
  presets: Record<string, Record<string, Params>>;
  limits: { max_upload_mb: number; work_side: number };
}

export interface ImageInfo {
  id: string;
  filename: string;
  category: string;
  width: number;
  height: number;
  preview: string;
  histogram: number[];
}

export interface StepImage {
  label: string;
  image: string;
}

export interface ProcessResponse {
  image: string;
  params: Params;
  duration_ms: number;
  histogram: number[];
  steps?: StepImage[];
  original?: string;
  size: [number, number];
}

export interface CompareResponse {
  items: StepImage[];
  duration_ms: number;
}

export interface ResultItem {
  id: string;
  effect: string;
  effect_name: string;
  params: Params;
  category: string;
  filename: string | null;
  width: number;
  height: number;
  duration_ms: number;
  created_at: number;
  image_id: string | null;
  image_available: boolean;
  thumb_url: string;
  image_url: string;
}

export interface ResultsPage {
  items: ResultItem[];
  total: number;
  page: number;
  per_page: number;
}

export interface Preset {
  id: number;
  name: string;
  effect: string;
  params: Params;
  created_at: number;
}

export interface Sample {
  id: string;
  category: string;
  name: string;
  thumb_url: string;
}

export type Tab = "compare" | "steps" | "analysis" | "history";
