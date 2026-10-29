# Latency Report (n=60 E2E queries, n=60 retrieval queries)

- Hardware: Apple M2, 9 GB RAM, arm64, MPS
- Provider: OpenRouter -> z-ai/glm-5.3-flash (generator; primary judge same rotation head)

| Metric | E2E generation (s) | Retrieval only (ms, rerank off) |
|---|---:|---:|
| median | 14.24 | 94.5 |
| p95 | 57.92 | 125.6 |
| max | 626.81 | 9241.5 |

## Outliers (kept, not dropped)

- `1409.3215_Q09` — 82.93 s (E2E)
- `1407.7906_Q11` — 94.28 s (E2E)
- `1409.3215_Q15` — 626.81 s (E2E)

The max (626.81 s, `1409.3215_Q15`) is a provider-side stall on the free-model path; it is retained in all statistics. 