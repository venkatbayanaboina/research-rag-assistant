# Retrieval Benchmark — mode=dense, k=5, rerank=off

- Cases: 60 (gold QA pairs for 4 indexed papers)
- Hit@1: **53.33%**
- Recall@3: **70.00%**
- Recall@5: **80.00%**
- MRR: **0.6356**
- Latency/query: mean 251.8 ms · p50 82.8 ms · p95 114.0 ms (embedding + search, model warm)

| Type | n | Hit@1 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| equation | 18 | 55.56% | 72.22% | 0.6083 |
| figure | 18 | 50.00% | 72.22% | 0.5972 |
| table | 6 | 50.00% | 100.00% | 0.7222 |
| text | 18 | 55.56% | 88.89% | 0.6722 |

## Missed cases (no evidence-page chunk in top-5)

- `1407.7906_Q03` (figure): In Figure 1, what three main actors are illustrated for an intermediate layer h in the generative model?
- `1407.7906_Q14` (equation): In the context of a linear top-level auto-encoder, what is the form of the optimal denoising reconstruction function min
- `1407.7906_Q15` (figure): What does Figure 5 illustrate regarding the architecture for structured output or multi-modal cases?
- `1408.5882_Q05` (figure): According to Figure 1, what are the main sequential layers in the model architecture after the sentence representation?
- `1408.5882_Q12` (equation): What does Equation (1) represent in the context of the model?
- `1408.5882_Q13` (equation): In Equation (2), ci = f(w · xi:i+h−1 + b), what does h represent?
- `1408.5882_Q15` (equation): What is the purpose of the operation described by c = [c1, c2, . . . , cn−h+1] in Equation (3)?
- `1409.3215_Q04` (text): How did the authors address the issue of wasted computation in minibatches due to varying sentence lengths?
- `1409.3215_Q05` (figure): According to Figure 1, what is the purpose of the special end-of-sentence token in the model?
- `1409.3215_Q06` (figure): What is the key observation about the input sentence processing shown in Figure 1 that helps the optimization problem?
- `1409.3215_Q15` (equation): What does Equation (2) aim to find, and what is the meaning of ˆT in this context?
- `1409.7770_Q05` (text): What is the primary limitation of the current experimental scheme regarding speedup, and what is planned for future stud