# Retrieval Benchmark — mode=hybrid, k=5, rerank=off

- Cases: 60 (gold QA pairs for 4 indexed papers)
- Hit@1: **60.00%**
- Recall@3: **86.67%**
- Recall@5: **91.67%**
- MRR: **0.7367**
- Latency/query: mean 246.9 ms · p50 94.5 ms · p95 125.6 ms (embedding + search, model warm)

| Type | n | Hit@1 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| equation | 18 | 66.67% | 83.33% | 0.7333 |
| figure | 18 | 55.56% | 94.44% | 0.7269 |
| table | 6 | 50.00% | 100.00% | 0.7083 |
| text | 18 | 61.11% | 94.44% | 0.7593 |

## Missed cases (no evidence-page chunk in top-5)

- `1407.7906_Q14` (equation): In the context of a linear top-level auto-encoder, what is the form of the optimal denoising reconstruction function min
- `1407.7906_Q15` (figure): What does Figure 5 illustrate regarding the architecture for structured output or multi-modal cases?
- `1408.5882_Q12` (equation): What does Equation (1) represent in the context of the model?
- `1409.3215_Q04` (text): How did the authors address the issue of wasted computation in minibatches due to varying sentence lengths?
- `1409.3215_Q15` (equation): What does Equation (2) aim to find, and what is the meaning of ˆT in this context?