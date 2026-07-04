# Part 1.3 — MoE Benchmark Analysis (TP vs EP)

Measured with `mpiexec -n 4 python benchmark.py --sweep` on a single 16-logical-core
Windows box (MS-MPI 10.1, NumPy CPU), `feature_dim=32, hidden_dim=128,
output_dim=64, topk=2, num_experts=4`. Average of 3 forward passes after warm-up.

| batch | simple ms | TP ms | EP ms | EP/TP |
|------:|----------:|------:|------:|------:|
| 8     | 0.245     | 5.03  | 0.53  | 0.10  |
| 32    | 1.85      | 13.65 | 0.95  | 0.07  |
| 128   | 6.12      | 50.82 | 3.44  | 0.07  |
| 512   | 23.82     | 205.4 | 12.85 | 0.06  |

**EP is ~10-16x faster than TP here.** The reason is entirely in the
communication pattern of each design as implemented:

- **Tensor Parallel (TP).** Every rank holds a *shard* of *every* expert. To
  evaluate one expert we must reassemble its full output, which requires an
  all-gather inside `ShardedLinear`. Because the MoE combination loops over
  tokens and top-k slots, TP triggers `O(batch x topk)` separate all-gather
  calls, each moving only a tiny slice. Communication latency (not bandwidth)
  dominates, so runtime scales linearly with `batch x topk` and the many small
  messages crush throughput. This is why TP time explodes from 5 ms to 205 ms as
  batch grows.

- **Expert Parallel (EP).** Each rank hosts one whole expert. A rank runs its
  expert on the tokens routed to it, and a **single** all-gather of the
  full-batch output buffers lets every rank assemble the final result. One large
  collective replaces thousands of small ones, so EP tracks the single-process
  baseline closely and even beats it at large batch (work is split across ranks).

**Takeaway.** For MoE, expert parallelism is the natural fit: experts are
independent, routing is sparse, and a single all-to-all / all-gather per layer
suffices. Tensor parallelism shines for *dense* giant matmuls (a single huge
GEMM split across devices with one reduction), but applying it per-expert here
multiplies the number of collectives and is the wrong tool. A production TP-MoE
would batch all tokens for an expert and shard the *big* GEMM once — not
all-gather per token — which would remove most of the overhead measured above.
The naive per-token implementation makes the latency cost of fine-grained
communication vividly visible, which is the point of the exercise.
