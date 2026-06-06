import { describe, expect, it, afterEach } from "vitest";
import { memoryConfigSchema } from "./config.js";

describe("m2-memory-engine config", () => {
  const minValid = {
    qdrant: { url: "http://qdrant:6333" },
    embeddings: { url: "http://bge:8000/v1", apiKey: "sk-test" },
  };

  it("accepts minimal required config and applies defaults", () => {
    const cfg = memoryConfigSchema.parse(minValid);
    expect(cfg.qdrant.url).toBe("http://qdrant:6333");
    expect(cfg.qdrant.collection).toBe("agent_memory_default");
    expect(cfg.qdrant.apiKey).toBeUndefined();
    expect(cfg.qdrant.perAgentCollections).toBeUndefined();
    expect(cfg.embeddings.model).toBe("bge-m3");
    expect(cfg.embeddings.dimensions).toBe(1024);
    expect(cfg.autoRecall.enabled).toBe(true);
    expect(cfg.autoRecall.maxResults).toBe(5);
    expect(cfg.autoRecall.minScore).toBe(0.3);
    expect(cfg.autoCapture.enabled).toBe(false);
    expect(cfg.autoCapture.maxChars).toBe(500);
    expect(cfg.autoCapture.dedupThreshold).toBe(0.95);
  });

  it("accepts full config including perAgentCollections", () => {
    const cfg = memoryConfigSchema.parse({
      qdrant: {
        url: "http://qdrant:6333",
        collection: "custom_col",
        apiKey: "qdrant-key",
        perAgentCollections: { accounting: "agent_accounting", domek: "agent_domek" },
      },
      embeddings: {
        url: "http://bge:8000/v1",
        apiKey: "sk-test",
        model: "bge-m3",
        dimensions: 512,
      },
      memgraph: { url: "bolt://mg:7687", enabled: true },
      ner: { url: "http://ner:8080", enabled: true },
      autoRecall: {
        enabled: false,
        maxResults: 10,
        minScore: 0.5,
        includeGraph: true,
        graphHops: 3,
      },
      autoCapture: { enabled: true, maxChars: 1000, extractEntities: true, dedupThreshold: 0.9 },
    });

    expect(cfg.qdrant.collection).toBe("custom_col");
    expect(cfg.qdrant.perAgentCollections).toEqual({
      accounting: "agent_accounting",
      domek: "agent_domek",
    });
    expect(cfg.embeddings.dimensions).toBe(512);
    expect(cfg.memgraph.enabled).toBe(true);
    expect(cfg.ner.enabled).toBe(true);
    expect(cfg.autoRecall.enabled).toBe(false);
    expect(cfg.autoRecall.graphHops).toBe(3);
    expect(cfg.autoCapture.enabled).toBe(true);
    expect(cfg.autoCapture.maxChars).toBe(1000);
  });

  it("throws when qdrant.url is missing", () => {
    expect(() =>
      memoryConfigSchema.parse({
        qdrant: {},
        embeddings: { url: "http://bge:8000/v1", apiKey: "sk-test" },
      }),
    ).toThrow("qdrant.url is required");
  });

  it("throws when embeddings.url is missing", () => {
    expect(() =>
      memoryConfigSchema.parse({
        qdrant: { url: "http://qdrant:6333" },
        embeddings: { apiKey: "sk-test" },
      }),
    ).toThrow("embeddings.url and embeddings.apiKey are required");
  });

  it("throws when embeddings.apiKey is missing", () => {
    expect(() =>
      memoryConfigSchema.parse({
        qdrant: { url: "http://qdrant:6333" },
        embeddings: { url: "http://bge:8000/v1" },
      }),
    ).toThrow("embeddings.url and embeddings.apiKey are required");
  });

  it("throws on unknown top-level keys", () => {
    expect(() => memoryConfigSchema.parse({ ...minValid, unexpected: true })).toThrow(
      "unknown keys: unexpected",
    );
  });

  it("throws on unknown qdrant keys", () => {
    expect(() =>
      memoryConfigSchema.parse({
        qdrant: { url: "http://qdrant:6333", unknown: true },
        embeddings: { url: "http://bge:8000/v1", apiKey: "sk-test" },
      }),
    ).toThrow("unknown keys: unknown");
  });

  it("rejects autoCapture.maxChars below 100", () => {
    expect(() => memoryConfigSchema.parse({ ...minValid, autoCapture: { maxChars: 50 } })).toThrow(
      "autoCapture.maxChars must be between 100 and 10000",
    );
  });

  it("rejects autoCapture.maxChars above 10000", () => {
    expect(() =>
      memoryConfigSchema.parse({ ...minValid, autoCapture: { maxChars: 99999 } }),
    ).toThrow("autoCapture.maxChars must be between 100 and 10000");
  });

  it("expands env vars in url and apiKey fields", () => {
    const prev = process.env.TEST_QDRANT_URL;
    const prevKey = process.env.TEST_EMB_KEY;
    try {
      process.env.TEST_QDRANT_URL = "http://real-qdrant:6333";
      process.env.TEST_EMB_KEY = "real-key-123";
      const cfg = memoryConfigSchema.parse({
        qdrant: { url: "${TEST_QDRANT_URL}" },
        embeddings: { url: "http://bge:8000/v1", apiKey: "${TEST_EMB_KEY}" },
      });
      expect(cfg.qdrant.url).toBe("http://real-qdrant:6333");
      expect(cfg.embeddings.apiKey).toBe("real-key-123");
    } finally {
      if (prev === undefined) delete process.env.TEST_QDRANT_URL;
      else process.env.TEST_QDRANT_URL = prev;
      if (prevKey === undefined) delete process.env.TEST_EMB_KEY;
      else process.env.TEST_EMB_KEY = prevKey;
    }
  });

  it("throws when referenced env var is not set", () => {
    delete process.env.MISSING_VAR_12345;
    expect(() =>
      memoryConfigSchema.parse({
        qdrant: { url: "${MISSING_VAR_12345}" },
        embeddings: { url: "http://bge:8000/v1", apiKey: "sk-test" },
      }),
    ).toThrow("MISSING_VAR_12345");
  });

  it("ignores perAgentCollections when null or array", () => {
    const withNull = memoryConfigSchema.parse({
      ...minValid,
      qdrant: { url: "http://qdrant:6333", perAgentCollections: null },
    });
    expect(withNull.qdrant.perAgentCollections).toBeUndefined();

    const withArray = memoryConfigSchema.parse({
      ...minValid,
      qdrant: { url: "http://qdrant:6333", perAgentCollections: ["a"] },
    });
    expect(withArray.qdrant.perAgentCollections).toBeUndefined();
  });

  it("throws on non-object config", () => {
    expect(() => memoryConfigSchema.parse(null)).toThrow("config required");
    expect(() => memoryConfigSchema.parse("string")).toThrow("config required");
    expect(() => memoryConfigSchema.parse([])).toThrow("config required");
  });
});
