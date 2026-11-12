import { useQuery } from "@tanstack/react-query";

export type ShapeDim = [string, number];
export type GNode = {
  id: string; label: string; kind: string; level: "macro" | "group" | "block" | "op"; parent: string | null;
  inputs: Record<string, ShapeDim[]>; outputs: Record<string, ShapeDim[]>; weights: Record<string, ShapeDim[]>;
  params: number; flops: number; op: string; eq: string; info: string; ref: string; meta: Record<string, any>;
};
export type Graph = {
  nodes: GNode[]; edges: { source: string; target: string; label: string }[];
  context: { T: number; n_text: number; n_vision: number; vision: any; img_hw: number[] | null };
  totals: { prefill_flops: number };
  summary?: { params_total: number; params_active: number; text: any; vision: any; costs: any; spec: Spec };
};
export type Spec = {
  family: string; architecture: string;
  text: {
    num_layers: number; hidden_size: number; intermediate_size: number | null; num_q_heads: number; num_kv_heads: number;
    head_dim: number; rotary_dim: number; vocab_size: number; max_position_embeddings: number; tie_word_embeddings: boolean;
    attn_output_gate: boolean; layer_types: string[]; rope: any; mtp_layers: number;
    moe: null | { num_experts: number; top_k: number; expert_ffn: number; shared_ffn: number };
    gdn: null | { num_k_heads: number; num_v_heads: number; k_head_dim: number; v_head_dim: number; conv_kernel: number };
  };
  vision: null | {
    depth: number; hidden_size: number; intermediate_size: number; num_heads: number; head_dim: number; patch_size: number;
    spatial_merge_size: number; temporal_patch_size: number; out_hidden_size: number; num_position_embeddings: number;
    deepstack_visual_indexes: number[]; hidden_act: string;
  };
  params: { total: number; active: number; total_with_mtp: number; text: any; vision: any };
  costs: any; preprocessor: any; generation: any; tokens: any;
};
export type ModelRow = { key: string; hf_id: string; family: string; label: string; released: string; license: string; variant: string; tier: string; notes: string[]; spec: Spec };
export type Specs = { models: ModelRow[]; families: Record<string, { label: string; date: string; paper: string }>; lineage: { date: string; name: string; change: string }[] };
export type TraceRow = { id: string; model: string; family: string; sample: string; mode: string; intervention: string; T: number; n_vision_tokens: number; bytes: number; policy: string; demo?: boolean };
export type Arr = { shape: number[]; dtype: string; data: number[] };

export async function get<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

const q = <T,>(key: unknown[], url: string | null, opts: { staleTime?: number } = {}) =>
  useQuery<T>({ queryKey: key, queryFn: () => get<T>(url!), enabled: !!url, staleTime: opts.staleTime ?? Infinity, retry: false });

export const useSpecs = () => q<Specs>(["specs"], "/api/specs");
export const useGraph = (model: string | null, params: Record<string, string | number | boolean> = {}) => {
  const s = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)])).toString();
  return q<Graph>(["graph", model, s], model ? `/api/graph/${model}?${s}` : null);
};
export const useTraces = () => q<TraceRow[]>(["traces"], "/api/traces", { staleTime: 10_000 });
export const useTracePart = <T = any,>(tid: string | null, part: string) => q<T>(["trace", tid, part], tid ? `/api/traces/${tid}/${part}` : null);
export const usePreview = (tid: string | null, key: string | null, opts: { row?: number; rows?: string; stride?: number } = {}) => {
  const s = new URLSearchParams(Object.entries(opts).filter(([, v]) => v !== undefined).map(([k, v]) => [k, String(v)])).toString();
  return q<Arr>(["preview", tid, key, s], tid && key ? `/api/traces/${tid}/preview/${key}?${s}` : null);
};
export const useInterventions = () => q<{ name: string; description: string; families: string[] }[]>(["ivs"], "/api/interventions");
export const useTraining = () => q<any[]>(["training"], "/api/training", { staleTime: 10_000 });
export const useTrainingFile = <T = any,>(rid: string | null, name: string | null) =>
  q<T>(["trainfile", rid, name], rid && name ? `/api/training/${rid}/${name}` : null);

/** row-major helpers */
export const at2 = (a: Arr, i: number, j: number) => a.data[i * a.shape[1] + j];
export function rowOf(a: Arr, i: number): number[] {
  const n = a.shape.slice(1).reduce((x, y) => x * y, 1);
  return a.data.slice(i * n, (i + 1) * n);
}
