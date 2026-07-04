import argparse
import json
import math


def model_training_cost_analysis_llama(model_config_path):
    """Estimate params / per-layer forward TFLOPs / peak fp16 memory for Llama-7B.

    Returns
    -------
    total_params : int
        Word-embedding + (Llama uses RoPE, so no learned positional embedding) +
        all transformer layers + final norm + LM head.
    flops_layer_TF : float
        Forward FLOPs of a single transformer layer (in TFLOPs) at seq_len =
        max_position_embeddings, batch = 1.
    peak_memory_GB : float
        Peak activation memory (fp16, 2 bytes) for the forward pass of one
        transformer layer, assuming checkpoint rematerialization at each layer
        boundary (so only one layer's activations are live at a time).
    """
    with open(model_config_path, "r") as f:
        cfg = json.load(f)

    h = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    n_layers = cfg["num_hidden_layers"]
    vocab = cfg["vocab_size"]
    seq = cfg.get("max_position_embeddings", cfg.get("max_sequence_length", 2048))

    # ---- Parameters ----
    # Attention (Llama): W_q, W_k, W_v, W_o each h x h  -> 4 * h*h
    attn_params = 4 * h * h
    # MLP (SwiGLU): gate, up (h x inter) and down (inter x h) -> 3 * h * inter
    mlp_params = 3 * h * inter
    # 2 RMSNorm weights per layer, each of size h.
    norm_params = 2 * h
    per_layer = attn_params + mlp_params + norm_params

    embedding = vocab * h                 # word embeddings
    # RoPE => no learned positional embeddings.
    final_norm = h
    lm_head = vocab * h                    # tie_word_embeddings is false for Llama-7B
    total_params = embedding + n_layers * per_layer + final_norm + lm_head

    # ---- Forward FLOPs of ONE transformer layer (batch=1, seq=S) ----
    # matmul of (S x p) by (p x q) costs 2*S*p*q FLOPs.
    S = seq
    # Attention projections: q,k,v,o -> 4 * (2 * S * h * h)
    flops_attn_proj = 4 * (2 * S * h * h)
    # Attention scores QK^T: 2 * S * S * h ; and softmax@V: 2 * S * S * h
    flops_attn_core = 2 * (2 * S * S * h)
    # MLP: gate + up (2 * 2*S*h*inter) + down (2*S*inter*h) = 3 * 2*S*h*inter
    flops_mlp = 3 * (2 * S * h * inter)
    flops_layer = flops_attn_proj + flops_attn_core + flops_mlp
    flops_layer_TF = flops_layer / 1e12

    # ---- Peak activation memory for one layer's forward (fp16, 2 bytes) ----
    # Dominant terms held for one layer: the S x S attention scores (per head we
    # store S*S; summed over heads = n_heads * S * S) plus the S x inter MLP
    # hidden activations, plus a few S x h buffers.
    n_heads = cfg["num_attention_heads"]
    bytes_fp16 = 2
    act_scores = n_heads * S * S            # attention probability matrix
    act_mlp = S * inter                     # SwiGLU hidden
    act_hidden = 6 * S * h                  # q,k,v,attn_out,residual,norm buffers
    peak_elems = act_scores + act_mlp + act_hidden
    peak_memory_GB = peak_elems * bytes_fp16 / (1024 ** 3)

    return total_params, flops_layer_TF, peak_memory_GB


def model_training_cost_analysis_deepseek(model_config_path):
    """Estimate params / per-layer forward TFLOPs / peak fp16 memory for DeepSeek-V3.

    DeepSeek-V3 uses Multi-head Latent Attention (MLA) with LoRA-compressed
    q/kv projections and a MoE FFN (n_routed_experts, num_experts_per_tok active
    + n_shared_experts). We count *total* params (all experts) but per-token
    FLOPs use only the *activated* experts, which is the point of MoE.
    """
    with open(model_config_path, "r") as f:
        cfg = json.load(f)

    h = cfg["hidden_size"]
    n_layers = cfg["num_hidden_layers"]
    vocab = cfg["vocab_size"]
    seq = cfg.get("max_position_embeddings", 4096)

    # MLA dims
    q_lora = cfg["q_lora_rank"]
    kv_lora = cfg["kv_lora_rank"]
    qk_nope = cfg["qk_nope_head_dim"]
    qk_rope = cfg["qk_rope_head_dim"]
    v_head = cfg["v_head_dim"]
    n_heads = cfg["num_attention_heads"]
    qk_head = qk_nope + qk_rope

    # ---- MLA attention params (per layer) ----
    # q: down h->q_lora, up q_lora->n_heads*qk_head
    q_params = h * q_lora + q_lora * (n_heads * qk_head)
    # kv: down h->(kv_lora+qk_rope), up kv_lora->n_heads*(qk_nope+v_head)
    kv_params = h * (kv_lora + qk_rope) + kv_lora * (n_heads * (qk_nope + v_head))
    # output proj: n_heads*v_head -> h
    o_params = (n_heads * v_head) * h
    attn_params = q_params + kv_params + o_params

    # ---- MoE FFN params (per MoE layer) ----
    moe_inter = cfg["moe_intermediate_size"]
    n_routed = cfg["n_routed_experts"]
    n_shared = cfg["n_shared_experts"]
    topk = cfg["num_experts_per_tok"]
    dense_inter = cfg["intermediate_size"]
    first_k_dense = cfg["first_k_dense_replace"]

    # A SwiGLU expert has 3 * h * inter params.
    routed_experts_params = n_routed * (3 * h * moe_inter)
    shared_experts_params = n_shared * (3 * h * moe_inter)
    router_params = h * n_routed
    moe_ffn_params = routed_experts_params + shared_experts_params + router_params
    dense_ffn_params = 3 * h * dense_inter

    norm_params = 2 * h

    # first_k_dense_replace layers use a dense MLP; the rest use MoE.
    dense_layers = first_k_dense
    moe_layers = n_layers - first_k_dense
    total_layer_params = (
        dense_layers * (attn_params + dense_ffn_params + norm_params)
        + moe_layers * (attn_params + moe_ffn_params + norm_params)
    )

    embedding = vocab * h
    final_norm = h
    lm_head = vocab * h
    total_params = embedding + total_layer_params + final_norm + lm_head

    # ---- Forward FLOPs of one MoE transformer layer (batch=1, seq=S) ----
    S = seq
    # MLA projections (down/up q, down/up kv, output).
    flops_attn = 2 * S * (
        h * q_lora + q_lora * (n_heads * qk_head)
        + h * (kv_lora + qk_rope) + kv_lora * (n_heads * (qk_nope + v_head))
        + (n_heads * v_head) * h
    )
    # Attention core (scores + context): 2 * (2 * S * S * n_heads * qk/v head dims)
    flops_attn += 2 * S * S * n_heads * qk_head       # QK^T
    flops_attn += 2 * S * S * n_heads * v_head        # attn @ V
    # MoE: only topk routed experts + shared experts are active per token.
    active_experts = topk + n_shared
    flops_moe = active_experts * (2 * S * (3 * h * moe_inter))
    flops_layer = flops_attn + flops_moe
    flops_layer_TF = flops_layer / 1e12

    # ---- Peak fp16 activation memory for one MoE layer ----
    bytes_fp16 = 2
    act_scores = n_heads * S * S
    act_mlp = active_experts * S * moe_inter
    act_hidden = 8 * S * h
    peak_elems = act_scores + act_mlp + act_hidden
    peak_memory_GB = peak_elems * bytes_fp16 / (1024 ** 3)

    return total_params, flops_layer_TF, peak_memory_GB


def get_optimal_N_D_from_cost(cost_budget):
    """
    Given a $ training budget, pick the most cost-effective GPU, convert the
    budget to effective training FLOPs (MFU 40%), and solve the Chinchilla-style
    scaling law for the (N, D) that minimize loss under C = 6*N*D.

        L(N, D) = 406.4 / N^0.34 + 410.7 / D^0.29 + 1.69

    Returns
    -------
    N : float   Optimal total model parameters.
    D : float   Optimal number of training tokens.
    training_budget_flops : float   Effective training FLOPs.
    best_gpu : str  One of 'A100', 'V100', 'T4'.
    """
    mfu = 0.40
    gpus = {
        # name: (cost_per_hour, peak_fp16_TFLOPs)
        "A100": (4.0, 312),
        "V100": (2.5, 125),
        "T4":   (1.0, 65),
    }

    # Effective FLOPs per dollar = (peak_TFLOPs*1e12 * mfu * 3600 s) / cost_per_hr.
    best_gpu = None
    best_flops_per_dollar = -1.0
    for name, (cph, tflops) in gpus.items():
        flops_per_dollar = (tflops * 1e12 * mfu * 3600) / cph
        if flops_per_dollar > best_flops_per_dollar:
            best_flops_per_dollar = flops_per_dollar
            best_gpu = name

    training_budget_flops = best_flops_per_dollar * cost_budget

    # Compute-optimal allocation. With C = 6 N D and the given exponents, minimize
    # L subject to fixed C. Grid-search N (log space); D = C / (6 N).
    C = training_budget_flops
    a, alpha = 406.4, 0.34
    b, beta = 410.7, 0.29

    best_N, best_D, best_L = None, None, float("inf")
    # Search N over a wide log range.
    for logN in [x / 100.0 for x in range(600, 1400)]:  # 1e6 .. 1e14
        N = 10 ** logN
        D = C / (6 * N)
        if D <= 0:
            continue
        L = a / (N ** alpha) + b / (D ** beta) + 1.69
        if L < best_L:
            best_L, best_N, best_D = L, N, D

    return best_N, best_D, training_budget_flops, best_gpu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model training cost analysis')
    parser.add_argument('--model_config', type=str, help='Path to model config file')
    parser.add_argument('--training_budget', type=float, default=None, help='Training budget')
    args = parser.parse_args()

    if args.model_config:
        if 'deepseek' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_deepseek(args.model_config)
        elif 'llama' in args.model_config:
            num_parameters, num_flops, memory_cost = model_training_cost_analysis_llama(args.model_config)
        else:
            print('Unknown LLM Type!')
            exit()
        print(f"Number of parameters: {num_parameters}")
        print(f"Number of TFLOPs: {num_flops}")
        print(f"Peak memory cost: {memory_cost} GBs")

    if args.training_budget:
        N, D, training_budget_flops, best_gpu = get_optimal_N_D_from_cost(args.training_budget)
        print(f"best_gpu: {best_gpu}")
        print(f"training_budget_flops: {training_budget_flops}")
        print(f"Optimal N: {N}")
        print(f"Optimal D: {D}")
