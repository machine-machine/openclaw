export type QdrantConfig = {
  url: string;
  collection: string;
  apiKey?: string;
};

export type EmbeddingsConfig = {
  url: string;
  apiKey: string;
  model: string;
  dimensions: number;
};

export type MemgraphConfig = {
  url: string;
  enabled: boolean;
};

export type NerConfig = {
  url: string;
  enabled: boolean;
};

export type AutoRecallConfig = {
  enabled: boolean;
  maxResults: number;
  minScore: number;
  includeGraph: boolean;
  graphHops: number;
};

export type AutoCaptureConfig = {
  enabled: boolean;
  maxChars: number;
  extractEntities: boolean;
  dedupThreshold: number;
};

export type MemoryConfig = {
  qdrant: QdrantConfig;
  embeddings: EmbeddingsConfig;
  memgraph: MemgraphConfig;
  ner: NerConfig;
  autoRecall: AutoRecallConfig;
  autoCapture: AutoCaptureConfig;
};

export const MEMORY_CATEGORIES = ["preference", "fact", "decision", "entity", "other"] as const;
export type MemoryCategory = (typeof MEMORY_CATEGORIES)[number];

export const DEFAULT_COLLECTION = "agent_memory_default";
export const DEFAULT_DIMENSIONS = 1024;
export const DEFAULT_MODEL = "bge-m3";
export const DEFAULT_CAPTURE_MAX_CHARS = 500;
export const DEFAULT_DEDUP_THRESHOLD = 0.95;

function resolveEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, envVar) => {
    const envValue = process.env[envVar];
    if (!envValue) {
      throw new Error(`Environment variable ${envVar} is not set`);
    }
    return envValue;
  });
}

function assertAllowedKeys(value: Record<string, unknown>, allowed: string[], label: string) {
  const unknown = Object.keys(value).filter((key) => !allowed.includes(key));
  if (unknown.length === 0) {
    return;
  }
  throw new Error(`${label} has unknown keys: ${unknown.join(", ")}`);
}

export const memoryConfigSchema = {
  parse(value: unknown): MemoryConfig {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error("m2-memory-engine: config required");
    }
    const cfg = value as Record<string, unknown>;
    assertAllowedKeys(
      cfg,
      ["qdrant", "embeddings", "memgraph", "ner", "autoRecall", "autoCapture"],
      "m2-memory-engine config",
    );

    // Qdrant config (required)
    const qdrant = cfg.qdrant as Record<string, unknown> | undefined;
    if (!qdrant || typeof qdrant.url !== "string") {
      throw new Error("qdrant.url is required");
    }
    assertAllowedKeys(qdrant, ["url", "collection", "apiKey"], "qdrant config");

    // Embeddings config (required)
    const embeddings = cfg.embeddings as Record<string, unknown> | undefined;
    if (
      !embeddings ||
      typeof embeddings.url !== "string" ||
      typeof embeddings.apiKey !== "string"
    ) {
      throw new Error("embeddings.url and embeddings.apiKey are required");
    }
    assertAllowedKeys(embeddings, ["url", "apiKey", "model", "dimensions"], "embeddings config");

    // Memgraph config (optional)
    const memgraph = (cfg.memgraph as Record<string, unknown>) || {};
    if (cfg.memgraph) {
      assertAllowedKeys(memgraph, ["url", "enabled"], "memgraph config");
    }

    // NER config (optional)
    const ner = (cfg.ner as Record<string, unknown>) || {};
    if (cfg.ner) {
      assertAllowedKeys(ner, ["url", "enabled"], "ner config");
    }

    // AutoRecall config (optional)
    const autoRecall = (cfg.autoRecall as Record<string, unknown>) || {};
    if (cfg.autoRecall) {
      assertAllowedKeys(
        autoRecall,
        ["enabled", "maxResults", "minScore", "includeGraph", "graphHops"],
        "autoRecall config",
      );
    }

    // AutoCapture config (optional)
    const autoCapture = (cfg.autoCapture as Record<string, unknown>) || {};
    if (cfg.autoCapture) {
      assertAllowedKeys(
        autoCapture,
        ["enabled", "maxChars", "extractEntities", "dedupThreshold"],
        "autoCapture config",
      );
    }

    const captureMaxChars =
      typeof autoCapture.maxChars === "number" ? Math.floor(autoCapture.maxChars) : undefined;
    if (
      typeof captureMaxChars === "number" &&
      (captureMaxChars < 100 || captureMaxChars > 10_000)
    ) {
      throw new Error("autoCapture.maxChars must be between 100 and 10000");
    }

    return {
      qdrant: {
        url: resolveEnvVars(qdrant.url as string),
        collection:
          typeof qdrant.collection === "string"
            ? resolveEnvVars(qdrant.collection)
            : DEFAULT_COLLECTION,
        apiKey: typeof qdrant.apiKey === "string" ? resolveEnvVars(qdrant.apiKey) : undefined,
      },
      embeddings: {
        url: resolveEnvVars(embeddings.url as string),
        apiKey: resolveEnvVars(embeddings.apiKey as string),
        model: typeof embeddings.model === "string" ? embeddings.model : DEFAULT_MODEL,
        dimensions:
          typeof embeddings.dimensions === "number" ? embeddings.dimensions : DEFAULT_DIMENSIONS,
      },
      memgraph: {
        url: typeof memgraph.url === "string" ? resolveEnvVars(memgraph.url) : "",
        enabled: memgraph.enabled === true,
      },
      ner: {
        url: typeof ner.url === "string" ? resolveEnvVars(ner.url) : "",
        enabled: ner.enabled === true,
      },
      autoRecall: {
        enabled: autoRecall.enabled !== false,
        maxResults: typeof autoRecall.maxResults === "number" ? autoRecall.maxResults : 5,
        minScore: typeof autoRecall.minScore === "number" ? autoRecall.minScore : 0.3,
        includeGraph: autoRecall.includeGraph === true,
        graphHops: typeof autoRecall.graphHops === "number" ? autoRecall.graphHops : 2,
      },
      autoCapture: {
        enabled: autoCapture.enabled === true,
        maxChars: captureMaxChars ?? DEFAULT_CAPTURE_MAX_CHARS,
        extractEntities: autoCapture.extractEntities === true,
        dedupThreshold:
          typeof autoCapture.dedupThreshold === "number"
            ? autoCapture.dedupThreshold
            : DEFAULT_DEDUP_THRESHOLD,
      },
    };
  },
};
