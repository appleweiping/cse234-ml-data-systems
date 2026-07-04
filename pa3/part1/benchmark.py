import numpy as np
from mpi4py import MPI
from rng import get_rng, rng_context, register_rng
from mpiwrapper import mpi
from moe import SimpleMoE, MoE_EP, MoE_TP
import time

# Example usage
def run_moe(
    moe_type="tp", 
    batch_size=8, 
    feature_dim=32, 
    hidden_dim=128, 
    output_dim=64, 
    num_experts=None,
    topk=2
):
    """
    Unified function to run different types of MoE models
    
    Args:
        moe_type: Type of MoE to run ("simple", "ep", or "tp")
        batch_size: Number of samples in the batch
        feature_dim: Dimension of input features
        hidden_dim: Hidden dimension for experts
        output_dim: Output dimension
        topk: Number of experts to route each input to
    """
    # Get number of experts based on MPI world size
    num_experts = mpi.get_size()
    
    # Generate input data
    np.random.seed(0)
    X = np.random.randn(batch_size, feature_dim)

    if moe_type != "simple":
        # Synchronize the input data across all processes
        if mpi.get_rank() == 0:
            X = get_rng().randn(batch_size, feature_dim)
        else:
            X = None
        X = mpi.comm.bcast(X, root=0)
    
    # Create appropriate MoE model
    model_class = {
        "simple": SimpleMoE,
        "ep": MoE_EP,
        "tp": MoE_TP
    }.get(moe_type, MoE_TP)
    
    moe = model_class(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_experts=num_experts,
        topk=topk
    )
    
    # Run forward pass
    # Warm up
    _ = moe(X)
    
    # Measure time
    N = 3
    start_time = time.time()
    for _ in range(N):
        outputs = moe(X)
    end_time = time.time()
    avg_duration_ms = 1000 * (end_time - start_time) / N
    
    # Print timing information
    if mpi.get_rank() == 0:
        print(f"Forward pass time for {moe_type} MoE: {avg_duration_ms} ms")

    return dict(
        outputs=outputs,
        avg_duration_ms=avg_duration_ms
    )
    
    
def benchmark_moe():
    # Test simple MoE
    simple_result = run_moe(moe_type="simple")
    print(f"Simple MoE: {simple_result['avg_duration_ms']} ms")

    # Test TP MoE
    tp_result = run_moe(moe_type="tp")
    print(f"TP MoE: {tp_result['avg_duration_ms']} ms")

    # Test EP MoE
    ep_result = run_moe(moe_type="ep")
    print(f"EP MoE: {ep_result['avg_duration_ms']} ms")


def benchmark_sweep():
    """Sweep batch size and compare TP vs EP (and the single-process baseline).

    hidden_dim and output_dim are multiples of the world size so TP sharding is
    valid for any launch size. Only rank 0 prints the table.
    """
    world = mpi.get_size()
    feature_dim = 32
    hidden_dim = 32 * world      # divisible by world -> valid TP sharding
    output_dim = 16 * world
    topk = 2

    batch_sizes = [8, 32, 128, 512]
    rows = []
    for bs in batch_sizes:
        simple = run_moe("simple", batch_size=bs, feature_dim=feature_dim,
                         hidden_dim=hidden_dim, output_dim=output_dim, topk=topk)
        tp = run_moe("tp", batch_size=bs, feature_dim=feature_dim,
                     hidden_dim=hidden_dim, output_dim=output_dim, topk=topk)
        ep = run_moe("ep", batch_size=bs, feature_dim=feature_dim,
                     hidden_dim=hidden_dim, output_dim=output_dim, topk=topk)
        rows.append((bs, simple["avg_duration_ms"], tp["avg_duration_ms"], ep["avg_duration_ms"]))

    if mpi.get_rank() == 0:
        print("\n" + "=" * 62)
        print(f"MoE forward-pass benchmark  (world_size={world}, "
              f"experts={world}, topk={topk})")
        print(f"feature_dim={feature_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
        print("=" * 62)
        print(f"{'batch':>6} | {'simple ms':>10} | {'TP ms':>8} | {'EP ms':>8} | {'EP/TP':>6}")
        print("-" * 62)
        for bs, s, tp, ep in rows:
            print(f"{bs:>6} | {s:>10.3f} | {tp:>8.3f} | {ep:>8.3f} | {ep/tp:>6.2f}")
        print("=" * 62)


if __name__ == "__main__":
    import sys
    if "--sweep" in sys.argv:
        benchmark_sweep()
    else:
        benchmark_moe()
