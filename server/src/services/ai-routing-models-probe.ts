import type { AiProviderRouting } from "@paperclipai/shared";
import { unprocessable } from "../errors.js";

const PROBE_TIMEOUT_MS = 15_000;

/** Mirrors the routing schema's model-list cap so a huge catalog cannot bloat the stored connection. */
export const AI_ROUTING_MODELS_PROBE_LIMIT = 200;

export function probeModelsUrl(route: AiProviderRouting): string {
  const base =
    route.kind === "openrouter"
      ? "https://openrouter.ai/api/v1"
      : (route.baseUrl ?? "").replace(/\/+$/, "");
  return `${base}/models`;
}

function headersFor(route: AiProviderRouting, apiKey: string): Record<string, string> {
  if (route.auth === "none") return {};
  if (route.auth === "api_key") return { "x-api-key": apiKey, "anthropic-version": "2023-06-01" };
  return { Authorization: `Bearer ${apiKey}` };
}

/** Parses an OpenAI-compatible `{"object":"list","data":[{"id":...}]}` model list. */
export function parseModelsResponse(payload: unknown): Array<{ id: string; label: string }> {
  const data =
    typeof payload === "object" && payload !== null && Array.isArray((payload as { data?: unknown }).data)
      ? (payload as { data: unknown[] }).data
      : [];
  const seen = new Set<string>();
  const models: Array<{ id: string; label: string }> = [];
  for (const entry of data) {
    const id =
      typeof entry === "object" && entry !== null && typeof (entry as { id?: unknown }).id === "string"
        ? (entry as { id: string }).id.trim()
        : "";
    if (!id || seen.has(id)) continue;
    seen.add(id);
    models.push({ id, label: id });
  }
  return models
    .sort((a, b) => a.id.localeCompare(b.id, "en", { numeric: true, sensitivity: "base" }))
    .slice(0, AI_ROUTING_MODELS_PROBE_LIMIT);
}

/** Lists models from a local or gateway endpoint (or OpenRouter) before the
 * connection is saved. The caller-supplied URL is constrained by the routing
 * schema (HTTPS except loopback, no embedded credentials) and credentials are
 * never sent through a redirect. */
export async function probeAiRoutingModels(
  route: AiProviderRouting,
  apiKey = "",
  request: typeof fetch = fetch,
  timeoutMs = PROBE_TIMEOUT_MS,
): Promise<Array<{ id: string; label: string }>> {
  if (route.kind === "bedrock")
    throw unprocessable("Bedrock does not expose an OpenAI-compatible model list. Enter model IDs manually.", { code: "ai_routing_models_probe_unsupported" });
  let response: Response;
  try {
    response = await request(probeModelsUrl(route), {
      redirect: "error",
      signal: AbortSignal.timeout(timeoutMs),
      headers: headersFor(route, apiKey),
    });
  } catch {
    throw unprocessable("Could not reach the endpoint. Check the URL and try again.", { code: "ai_routing_models_probe_unreachable" });
  }
  if (!response.ok) {
    await response.body?.cancel();
    throw unprocessable(
      response.status === 401 || response.status === 403
        ? "The endpoint rejected the credentials."
        : "The endpoint could not list models. Try again.",
      { code: response.status === 401 || response.status === 403 ? "ai_routing_models_probe_unauthorized" : "ai_routing_models_probe_failed" },
    );
  }
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    throw unprocessable("The endpoint did not return a JSON model list.", { code: "ai_routing_models_probe_invalid" });
  }
  return parseModelsResponse(payload);
}
