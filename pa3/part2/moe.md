# Part 2.3 — MoE Cost Analysis (DeepSeek-V3) and the Advantages of MoE

`model_training_cost_analysis_deepseek` in `model_training_cost_analysis.py`
reads `deepseek_v3_config.json` and accounts for the two things that make a
DeepSeek-V3 layer different from a dense Llama layer:

1. **Multi-head Latent Attention (MLA).** Instead of full `h×h` q/k/v/o
   projections, DeepSeek compresses q through a `q_lora_rank=1536` bottleneck and
   k/v through a `kv_lora_rank=512` bottleneck, with decoupled RoPE dims
   (`qk_rope_head_dim=64`). We count the down- and up-projection matrices
   explicitly.
2. **Mixture-of-Experts FFN.** Each MoE layer holds `n_routed_experts=256`
   experts plus `n_shared_experts=1`, each a SwiGLU MLP of width
   `moe_intermediate_size=2048`. The first `first_k_dense_replace=3` layers keep
   a dense MLP (`intermediate_size=18432`); the remaining 58 layers are MoE.

Measured output of our function:

| Model | Total params | Per-layer fwd TFLOPs | Peak fp16 act (GB) |
|---|---|---|---|
| DeepSeek-V3 | **671.0 B** | 2390.2 (at seq=163840) | 6423.1 |

The **671 B total parameters exactly matches DeepSeek's published figure**,
which validates the parameter accounting. The per-layer FLOPs and activation
memory are large only because the config's `max_position_embeddings` is 163840;
at a normal training sequence length they shrink by orders of magnitude.

## Why the same dense function does not directly apply

The dense `model_training_cost_analysis_llama` assumes a single `h×inter` MLP per
layer and full `h×h` attention. Applied blindly to DeepSeek it would (a)
massively overcount FLOPs, because it would run **all 256 experts** on every
token instead of the `num_experts_per_tok=8` (+1 shared) that are actually
activated, and (b) mis-model attention, since MLA's compressed projections are
much cheaper than dense `h×h`. Hence the separate `deepseek` function that
counts **total** params (all experts, for memory/storage) but only the
**activated** experts for per-token compute.

## Advantages of MoE

- **Decoupling capacity from compute.** DeepSeek-V3 stores 671 B parameters but
  activates only ~37 B per token (8 of 256 routed experts + 1 shared). You buy
  the representational capacity of a huge model while paying the FLOPs of a much
  smaller one — this is exactly why DeepSeek *claims* a ~$5M training run for a
  frontier-quality model.
- **Better loss per FLOP.** For a fixed compute budget, a sparsely-activated MoE
  reaches lower loss than a dense model of equal *active* size, because the extra
  (inactive) parameters still add specialization without adding per-token cost.
- **Specialization / interpretability.** Different experts learn different
  sub-distributions (syntax, code, math), and the router sends each token to the
  most relevant ones, improving quality on heterogeneous data.
- **Scaling knobs.** Capacity scales by adding experts (memory/comm bound)
  rather than making every matmul bigger (compute bound), which maps cleanly onto
  **expert parallelism** — each device hosts a subset of experts and tokens are
  shuffled with all-to-all (exactly what Part 1's `MoE_EP` implements).

The trade-off is systems complexity: all-to-all communication, load-balancing
(the `aux_loss_alpha` in the config), and high total memory to *hold* every
expert even though only a few run per token.
