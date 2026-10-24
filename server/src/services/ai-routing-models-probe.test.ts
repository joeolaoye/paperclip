import http from "node:http";
import type { AddressInfo } from "node:net";
import { afterAll, describe, expect, it, vi } from "vitest";
import { aiProviderRoutingSchema, type AiProviderRouting } from "@paperclipai/shared";
import {
  AI_ROUTING_MODELS_PROBE_LIMIT,
  parseModelsResponse,
  probeAiRoutingModels,
  probeModelsUrl,
} from "./ai-routing-models-probe.js";

function route(input: Record<string, unknown>): AiProviderRouting {
  return aiProviderRoutingSchema.parse(input);
}

describe("parseModelsResponse", () => {
  it("sorts, dedupes, and keeps valid OpenAI-style entries", () => {
    expect(
      parseModelsResponse({
        object: "list",
        data: [
          { id: "qwen3-coder:30b", object: "model" },
          { id: "llama3.2:latest", object: "model" },
          { id: "qwen3-coder:30b" },
          { id: "  " },
          { object: "model" },
          "not-an-object",
        ],
      }),
    ).toEqual([
      { id: "llama3.2:latest", label: "llama3.2:latest" },
      { id: "qwen3-coder:30b", label: "qwen3-coder:30b" },
    ]);
  });

  it("caps the list at the routing schema's model limit", () => {
    const data = Array.from({ length: AI_ROUTING_MODELS_PROBE_LIMIT + 50 }, (_, i) => ({
      id: `model-${String(i).padStart(4, "0")}`,
    }));
    const parsed = parseModelsResponse({ data });
    expect(parsed).toHaveLength(AI_ROUTING_MODELS_PROBE_LIMIT);
    expect(parsed[0]?.id).toBe("model-0000");
  });

  it("returns an empty list for payloads without a data array", () => {
    expect(parseModelsResponse({ error: { message: "nope" } })).toEqual([]);
    expect(parseModelsResponse(null)).toEqual([]);
  });
});

describe("probeModelsUrl", () => {
  it("trims trailing slashes from the configured base URL", () => {
    expect(
      probeModelsUrl(route({ kind: "local", protocol: "chat", baseUrl: "http://localhost:11434/v1///", auth: "none" })),
    ).toBe("http://localhost:11434/v1/models");
  });

  it("uses OpenRouter's official endpoint", () => {
    expect(probeModelsUrl(route({ kind: "openrouter", protocol: "chat" }))).toBe(
      "https://openrouter.ai/api/v1/models",
    );
  });
});

function captureRequest() {
  const calls: Array<{ url: string; init: RequestInit }> = [];
  const request = vi.fn(async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    return new Response(JSON.stringify({ object: "list", data: [{ id: "m" }] }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });
  return { request, calls };
}

describe("probeAiRoutingModels request shape", () => {
  it("sends a bearer token, no-redirect, and a timeout signal", async () => {
    const { request, calls } = captureRequest();
    const models = await probeAiRoutingModels(
      route({ kind: "gateway", protocol: "chat", baseUrl: "https://gw.example.com/v1", auth: "bearer" }),
      "gateway-key",
      request as unknown as typeof fetch,
    );
    expect(models).toEqual([{ id: "m", label: "m" }]);
    expect(calls[0]?.url).toBe("https://gw.example.com/v1/models");
    expect(calls[0]?.init.redirect).toBe("error");
    expect(calls[0]?.init.signal).toBeInstanceOf(AbortSignal);
    expect(calls[0]?.init.headers).toEqual({ Authorization: "Bearer gateway-key" });
  });

  it("uses x-api-key with the Anthropic version header for messages endpoints", async () => {
    const { request, calls } = captureRequest();
    await probeAiRoutingModels(
      route({ kind: "gateway", protocol: "messages", baseUrl: "https://gw.example.com", auth: "api_key" }),
      "gateway-key",
      request as unknown as typeof fetch,
    );
    expect(calls[0]?.init.headers).toEqual({
      "x-api-key": "gateway-key",
      "anthropic-version": "2023-06-01",
    });
  });

  it("sends no credentials for unauthenticated local endpoints", async () => {
    const { request, calls } = captureRequest();
    await probeAiRoutingModels(
      route({ kind: "local", protocol: "chat", baseUrl: "http://localhost:11434/v1", auth: "none" }),
      "",
      request as unknown as typeof fetch,
    );
    expect(calls[0]?.init.headers).toEqual({});
  });

  it("rejects Bedrock routing with an actionable error", async () => {
    await expect(
      probeAiRoutingModels(route({ kind: "bedrock", protocol: "bedrock", auth: "bearer", region: "us-east-1" }), "key"),
    ).rejects.toMatchObject({ status: 422, details: { code: "ai_routing_models_probe_unsupported" } });
  });

  it("maps endpoint rejection and transport failure to distinct codes", async () => {
    await expect(
      probeAiRoutingModels(
        route({ kind: "gateway", protocol: "chat", baseUrl: "https://gw.example.com/v1", auth: "bearer" }),
        "key",
        (async () => new Response(null, { status: 401 })) as unknown as typeof fetch,
      ),
    ).rejects.toMatchObject({ status: 422, details: { code: "ai_routing_models_probe_unauthorized" } });
    await expect(
      probeAiRoutingModels(
        route({ kind: "gateway", protocol: "chat", baseUrl: "https://gw.example.com/v1", auth: "bearer" }),
        "key",
        (async () => {
          throw new Error("connection refused");
        }) as unknown as typeof fetch,
      ),
    ).rejects.toMatchObject({ status: 422, details: { code: "ai_routing_models_probe_unreachable" } });
  });

  it("rejects non-JSON responses", async () => {
    await expect(
      probeAiRoutingModels(
        route({ kind: "local", protocol: "chat", baseUrl: "http://localhost:11434/v1", auth: "none" }),
        "",
        (async () => new Response("<html>login</html>", { status: 200 })) as unknown as typeof fetch,
      ),
    ).rejects.toMatchObject({ status: 422, details: { code: "ai_routing_models_probe_invalid" } });
  });
});

// The probe runs against operator-supplied model servers, so the real fetch
// path (socket, JSON decode, abort-on-timeout) is exercised against a local
// HTTP server rather than only the mocked transport above.
describe("probeAiRoutingModels against a real endpoint", () => {
  const servers: http.Server[] = [];

  function listen(handler: http.RequestListener): Promise<string> {
    const server = http.createServer(handler);
    servers.push(server);
    return new Promise((resolve) => {
      server.listen(0, "127.0.0.1", () => {
        const address = server.address() as AddressInfo;
        resolve(`http://127.0.0.1:${address.port}/v1`);
      });
    });
  }

  afterAll(async () => {
    for (const server of servers) server.closeAllConnections();
    await Promise.all(servers.map((server) => new Promise<void>((resolve) => server.close(() => resolve()))));
  });

  it("lists models from an Ollama-style endpoint over the real transport", async () => {
    const baseUrl = await listen((req, res) => {
      if (req.url === "/v1/models") {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ object: "list", data: [{ id: "qwen3-coder:30b" }, { id: "llama3.2:latest" }] }));
        return;
      }
      res.writeHead(404); res.end();
    });
    const models = await probeAiRoutingModels(
      route({ kind: "local", protocol: "chat", baseUrl, auth: "none" }),
    );
    expect(models.map((m) => m.id)).toEqual(["llama3.2:latest", "qwen3-coder:30b"]);
  });

  it("maps a real 401 to the unauthorized code", async () => {
    const baseUrl = await listen((req, res) => {
      res.writeHead(401); res.end();
    });
    await expect(
      probeAiRoutingModels(route({ kind: "local", protocol: "chat", baseUrl, auth: "bearer" }), "key"),
    ).rejects.toMatchObject({ status: 422, details: { code: "ai_routing_models_probe_unauthorized" } });
  });

  it("aborts a hanging endpoint while the request is in flight", async () => {
    const baseUrl = await listen(() => {});
    await expect(
      probeAiRoutingModels(route({ kind: "local", protocol: "chat", baseUrl, auth: "none" }), "", fetch, 50),
    ).rejects.toMatchObject({ status: 422, details: { code: "ai_routing_models_probe_unreachable" } });
  });
});
