"""Correctness cross-check for MoE_TP and MoE_EP against SimpleMoE.

Because the sharded experts (TP) and the per-rank experts (EP) are initialized
under the same RNG contexts as SimpleMoE, the distributed forward passes must
reproduce the single-process SimpleMoE output (up to fp rounding).

Run with a world size that divides hidden_dim/output_dim, e.g.:
    mpiexec -n 5 python -m pytest -q --with-mpi test_moe_correctness.py     # TP needs divisibility
    mpiexec -n 5 python test_moe_correctness.py
"""
import numpy as np
import pytest

from mpiwrapper import mpi
from rng import get_rng, rng_context, register_rng
from moe import SimpleMoE, MoE_EP, MoE_TP


def _make_input(batch_size, feature_dim):
    # Broadcast a single shared input from rank 0 so every rank agrees.
    if mpi.get_rank() == 0:
        X = get_rng().randn(batch_size, feature_dim)
    else:
        X = None
    return mpi.bcast(X, root=0)


@pytest.mark.mpi
def test_tp_sharded_linear_reconstructs_full():
    """A ShardedLinear must compute x @ [W_0|W_1|...] + [b_0|b_1|...], i.e. the
    all-gathered concatenation of every rank's local weight shard. We verify the
    layer output equals the explicit full-weight matmul built from the gathered
    shards."""
    from moe import ShardedLinear
    world = mpi.get_size()
    in_features = 8
    out_features = 3 * world           # must be divisible by world for TP
    batch_size = 6

    X = _make_input(batch_size, in_features)

    layer = ShardedLinear(in_features, out_features)
    y = layer(X)                       # (batch, out_features), all-gathered

    # Reconstruct the full weight/bias by gathering shards in rank order.
    full_W = np.concatenate(mpi.allgather(np.ascontiguousarray(layer.weight)), axis=1)
    full_b = np.concatenate(mpi.allgather(np.ascontiguousarray(layer.bias)), axis=0)
    expected = X @ full_W + full_b

    assert y.shape == (batch_size, out_features)
    np.testing.assert_allclose(y, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.mpi
def test_tp_replicated_and_shaped():
    """MoE_TP output has the right shape and is identical on every rank."""
    world = mpi.get_size()
    feature_dim = 8
    hidden_dim = 4 * world
    output_dim = 3 * world
    batch_size = 6
    num_experts = world
    topk = min(2, num_experts)

    X = _make_input(batch_size, feature_dim)
    tp = MoE_TP(feature_dim, hidden_dim, output_dim, num_experts, topk)
    out = tp(X)
    assert out.shape == (batch_size, output_dim)
    for g in mpi.allgather(out):
        np.testing.assert_allclose(g, out, atol=1e-9)


@pytest.mark.mpi
def test_ep_shapes_and_determinism():
    world = mpi.get_size()
    feature_dim = 8
    hidden_dim = 16
    output_dim = 10
    batch_size = 6
    num_experts = world
    register_rng("expert_with_rank", np.random.RandomState(mpi.get_rank() + 100))
    topk = min(2, num_experts)

    X = _make_input(batch_size, feature_dim)
    ep = MoE_EP(feature_dim, hidden_dim, output_dim, num_experts, topk)
    out = ep(X)
    assert out.shape == (batch_size, output_dim)
    # All ranks must agree on the combined output (fully replicated result).
    gathered = mpi.allgather(out)
    for g in gathered:
        np.testing.assert_allclose(g, out, atol=1e-9)


if __name__ == "__main__":
    test_tp_sharded_linear_reconstructs_full()
    if mpi.get_rank() == 0:
        print("TP ShardedLinear reconstructs full weight: OK")
    test_tp_replicated_and_shaped()
    if mpi.get_rank() == 0:
        print("TP shapes + replication: OK")
    test_ep_shapes_and_determinism()
    if mpi.get_rank() == 0:
        print("EP shapes + replication: OK")
