# M² Memory Engine — OpenClaw Context Engine Plugin

**Status:** SPEC  
**Priority:** High — Product differentiator  
**Owner:** m2  
**Target:** `@machinemachine/memory-engine` (OpenClaw plugin, kind: `memory`)

---

## Problem

Stock OpenClaw ships with `memory-lancedb` — a local file-based vector DB with simple auto-recall/capture. It works, but it's a toy compared to what M² already has running:

- **Qdrant** (shared, network, cross-agent) — production vector search
- **Memgraph** (knowledge graph) — entity relationships, multi-hop traversal
- **RLM** (recursive language model) — iterative reasoning over memories
- **GLiNER NER** — zero-shot multilingual entity extraction
- **BGE-M3** — embeddings (1024-dim, already on GPU)
- **Unified Search** — fan-out across all sources in parallel

Running `memory-lancedb` alongside this stack is redundant. The M² custom OpenClaw branch should ship its own memory plugin that wires into the existing infra.

## Solution

Build `@machinemachine/memory-engine` — a drop-in replacement for `memory-lancedb` that uses the M² memory stack via the OpenClaw plugin SDK.

### Architecture

```
┌─────────────────────────────────────────────────┐
│  OpenClaw Plugin: @machinemachine/memory-engine  │
├─────────────────────────────────────────────────┤
│                                                  │
│  before_agent_start → AUTO-RECALL                │
│    ├─ Qdrant vector search (fast, < 50ms)        │
│    ├─ Memgraph graph traversal (entities)        │
│    └─ Merge + rank + inject into context         │
│                                                  │
│  agent_end → AUTO-CAPTURE                        │
│    ├─ GLiNER entity extraction                   │
│    ├─ Qdrant store (vector + metadata)           │
│    ├─ Memgraph upsert (entities + relations)     │
│    └─ Dedup via similarity threshold             │
│                                                  │
│  Tools: memory_recall, memory_store, memory_forget│
│    └─ Proxy to Qdrant + Memgraph                 │
│                                                  │
│  CLI: openclaw ltm search/list/stats/graph       │
│    └─ Query both vector + graph stores           │
│                                                  │
└──────────────┬──────────────┬───────────────────┘
               │              │
     ┌─────────▼──┐    ┌──────▼──────┐
     │   Qdrant    │    │  Memgraph   │
     │ (vectors)   │    │  (graph)    │
     └─────────────┘    └─────────────┘
           │                   │
     ┌─────▼─────┐     ┌──────▼──────┐
     │  BGE-M3   │     │  GLiNER NER │
     │ (GPU emb) │     │  (GPU NER)  │
     └───────────┘     └─────────────┘
```

### Plugin Config Schema

```json5
{
  plugins: {
    slots: { memory: "m2-memory-engine" },
    entries: {
      "m2-memory-engine": {
        enabled: true,
        config: {
          qdrant: {
            url: "http://memory-qdrant:6333",
            collection: "agent_memory_${AGENT_ID}",
          },
          embeddings: {
            url: "https://bge-proxy.machinemachine.ai/v1",
            apiKey: "${BGE_PROXY_TOKEN}",
            model: "bge-m3",
            dimensions: 1024,
          },
          memgraph: {
            url: "bolt://memory-memgraph:7687",
            enabled: true,
          },
          ner: {
            url: "http://memory-ner:8080",
            enabled: true,
          },
          autoRecall: {
            enabled: true,
            maxResults: 5,
            minScore: 0.3,
            includeGraph: true, // also query Memgraph for related entities
            graphHops: 2, // traverse up to 2 hops in the graph
          },
          autoCapture: {
            enabled: true,
            maxChars: 500,
            extractEntities: true, // run GLiNER on captured text
            dedupThreshold: 0.95,
          },
        },
      },
    },
  },
}
```

### Auto-Recall Flow (before_agent_start)

1. **Embed the user prompt** via BGE-M3 (< 50ms on GPU)
2. **Vector search** Qdrant collection for top-5 similar memories
3. **If graph enabled:** Extract entities from prompt via GLiNER → query Memgraph for related entities within 2 hops → fetch those memories from Qdrant too
4. **Merge & rank** — deduplicate, sort by `(vector_score * 0.7 + graph_relevance * 0.3)`, cap at 5
5. **Format** into `<relevant-memories>` block with injection protection
6. **Return** as `prependContext` in the hook response

Total latency target: < 200ms (vector: 50ms, graph: 100ms, NER: 50ms — parallel where possible)

### Auto-Capture Flow (agent_end)

1. **Filter** user messages (skip agent output, skip too short/long, skip injected context)
2. **Check capture triggers** — same regex set as memory-lancedb + custom M² patterns
3. **Embed** via BGE-M3
4. **Dedup** — check Qdrant for > 0.95 similarity match
5. **Store** in Qdrant with metadata: `{category, importance, agent_id, session_id, timestamp}`
6. **If NER enabled:** Run GLiNER on text → extract entities → upsert into Memgraph with relationships:
   - `(Entity)-[:MENTIONED_IN]->(Memory)`
   - `(Entity)-[:RELATED_TO]->(Entity)` (co-occurrence within same message)
   - `(Person)-[:KNOWS]->(Person)` (when both mentioned together)

### Tools (replacements for memory-lancedb tools)

| Tool            | Backend                                         | Enhancement over memory-lancedb                            |
| --------------- | ----------------------------------------------- | ---------------------------------------------------------- |
| `memory_recall` | Qdrant vector search + optional graph expansion | Cross-agent memory (shared Qdrant), graph-expanded results |
| `memory_store`  | Qdrant + Memgraph                               | Entity extraction + graph storage                          |
| `memory_forget` | Qdrant delete + Memgraph cleanup                | GDPR: removes from both stores                             |

### CLI Commands

```bash
openclaw ltm search "query"         # Vector search
openclaw ltm graph "entity"         # Memgraph traversal
openclaw ltm stats                  # Memory count, entity count, graph size
openclaw ltm list --recent 10       # Recent memories
openclaw ltm export                 # Export all memories as JSON
```

## Differentiators vs Stock OpenClaw

| Feature               | memory-lancedb      | m2-memory-engine                    |
| --------------------- | ------------------- | ----------------------------------- |
| Storage               | Local LanceDB files | Network Qdrant (shared)             |
| Cross-agent memory    | ❌ Per-agent        | ✅ Shared collections               |
| Knowledge graph       | ❌                  | ✅ Memgraph (entities + relations)  |
| Entity extraction     | ❌                  | ✅ GLiNER (zero-shot, multilingual) |
| Graph-expanded recall | ❌                  | ✅ "What's related to X?" via graph |
| Multi-hop reasoning   | ❌                  | Available via RLM fallback tool     |
| GPU embeddings        | Via API proxy       | ✅ Direct BGE-M3 on GPU             |
| Scalability           | Single-node         | Distributed (Qdrant cluster-ready)  |

## Implementation Plan

### Phase 1: Core Plugin (MVP)

- Plugin scaffold with OpenClaw plugin SDK
- Qdrant integration (store, search, delete)
- BGE-M3 embeddings via proxy
- `before_agent_start` auto-recall hook
- `agent_end` auto-capture hook
- memory_recall / memory_store / memory_forget tools
- Replace `memory-lancedb` in fleet config
- **Deliverable:** Drop-in replacement, same UX, better backend

### Phase 2: Graph Integration

- Memgraph connection + Cypher queries
- GLiNER entity extraction on capture
- Graph-expanded recall in auto-recall hook
- `openclaw ltm graph` CLI command
- **Deliverable:** Graph-aware memory that understands entity relationships

### Phase 3: Advanced Features

- Cross-agent memory sharing policies (opt-in)
- Memory consolidation (periodic merge of similar memories)
- Importance decay (old memories fade unless reinforced)
- RLM integration for complex recall queries
- Memory analytics dashboard
- **Deliverable:** Production-grade enterprise memory system

## Technical Notes

- Plugin lives in `extensions/m2-memory-engine/` in the m2-custom branch
- Uses OpenClaw plugin SDK: `api.on("before_agent_start")`, `api.on("agent_end")`, `api.registerTool()`
- Config validation via `memoryConfigSchema.parse()` pattern (same as memory-lancedb)
- All network calls have timeouts and fallbacks — if Qdrant is down, degrade gracefully
- Memgraph + NER are optional (`enabled: false` to disable)
- Collection naming: `agent_memory_${agentId}` — same as existing Qdrant setup
- Plugin registers as `kind: "memory"` so it occupies the `plugins.slots.memory` slot

## Risk & Mitigation

| Risk                                 | Mitigation                                                           |
| ------------------------------------ | -------------------------------------------------------------------- |
| Qdrant/Memgraph unavailable          | Graceful degradation — log warning, skip recall/capture, don't crash |
| Auto-recall latency > 200ms          | Parallel queries, timeout at 500ms, skip graph if slow               |
| Memory pollution from auto-capture   | Conservative trigger patterns + dedup threshold + max 3 per turn     |
| Prompt injection via stored memories | Same `<relevant-memories>` sandboxing as memory-lancedb              |
