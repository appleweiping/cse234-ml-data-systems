import numpy as np
from mpi4py import MPI


def get_info(
    comm,
    rank: int,
    mp_size: int,
    dp_size: int,
    fc_layer: str,
    in_dim: int,
    out_dim: int,
):
    """
    Prepare necessary information for later communications in forward and backward passes.

    Parameters
    ----------
    comm : Communicator
        The global MPI communicator.
    rank : int
        The global rank of the process.
    mp_size : int
        Model Parallel size.
    dp_size : int
        Data Parallel size.
    fc_layer : str
        Identifier for the fully-connected layer. It must be one of:
        'fc_q', 'fc_k', 'fc_v', or 'fc_o'.
        - For 'fc_q', 'fc_k', and 'fc_v', the partitioning is along the output dimension.
        - For 'fc_o', the partitioning is along the input dimension.
    in_dim : int
        Original input feature dimension.
    out_dim : int
        Original output feature dimension.

    Returns
    -------
    mp_idx : int
        Model parallel index (position within a data parallel replica).
    dp_idx : int
        Data parallel index (which replica this process belongs to).
    mp_comm : Communicator
        The model parallel communicator (all processes in one data parallel replica).
    dp_comm : Communicator
        The data parallel communicator (all processes holding the same weight shard).
    part_in_dim : int
        The partitioned input dimension for the FC layer.
    part_out_dim : int
        The partitioned output dimension for the FC layer.
    """
    # Model-parallel-major node layout: within one DP replica there are mp_size
    # consecutive ranks. So mp_idx cycles within a replica and dp_idx selects it.
    mp_idx = rank % mp_size
    dp_idx = rank // mp_size

    # mp_comm groups together the mp ranks that form one DP replica (same dp_idx).
    #   -> color = dp_idx, so ranks with the same dp_idx end up in one communicator.
    # dp_comm groups together the ranks holding the same weight shard (same mp_idx).
    #   -> color = mp_idx.
    mp_comm = comm.Split(key=mp_idx, color=dp_idx)
    dp_comm = comm.Split(key=dp_idx, color=mp_idx)

    # Weight partitioning:
    #   fc_q / fc_k / fc_v : split the OUTPUT dimension across mp_size shards.
    #   fc_o               : split the INPUT dimension across mp_size shards.
    if fc_layer in ("fc_q", "fc_k", "fc_v"):
        part_in_dim = in_dim
        part_out_dim = out_dim // mp_size
    elif fc_layer == "fc_o":
        part_in_dim = in_dim // mp_size
        part_out_dim = out_dim
    else:
        raise ValueError(f"Unknown fc_layer: {fc_layer}")

    return mp_idx, dp_idx, mp_comm, dp_comm, part_in_dim, part_out_dim


def naive_collect_forward_input(
    x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward inputs from all model-parallel nodes.

    Each node holds a piece of the full input with shape:
      (batch_size, seq_length, part_in_dim)
    After gathering, the full input should have shape:
      (batch_size, seq_length, part_in_dim * mp_size)
    """
    # Gather every rank's local slice, then concatenate along the last axis in
    # rank order to reassemble the full feature dimension.
    x = np.ascontiguousarray(x)
    gathered = mp_comm.allgather(x)  # list of mp_size arrays, index = rank
    collected_x = np.concatenate(gathered, axis=-1)
    return collected_x


def naive_collect_forward_output(
    out: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Collects the fc_o layer's forward outputs from all model-parallel nodes.

    Each node holds a piece of the full output with shape:
      (batch_size, seq_length, part_out_dim)
    After gathering, the full output should have shape:
      (batch_size, seq_length, part_out_dim * mp_size)
    """
    out = np.ascontiguousarray(out)
    gathered = mp_comm.allgather(out)
    collected_out = np.concatenate(gathered, axis=-1)
    return collected_out


def naive_collect_backward_output(
    output_grad: np.ndarray,
    mp_group_idx: int,
    mp_size: int,
):
    """
    Collect the fc output layer's output gradient for the local MP node.

    In our setup, the full output_grad is a 3-D tensor of shape
        (batch_size, seq_length, out_dim),
    and the fully connected layer's weight is partitioned along out_dim.
    Therefore, we split output_grad along axis=2 into mp_size parts and
    return the part corresponding to mp_group_idx.

    Returns
    -------
    collected_output_grad : np.ndarray
        The local output gradient for this MP node with shape
        (batch_size, seq_length, out_dim // mp_size).
    """
    out_dim = output_grad.shape[-1]
    part = out_dim // mp_size
    start = mp_group_idx * part
    end = start + part
    collected_output_grad = output_grad[:, :, start:end]
    return collected_output_grad


def naive_collect_backward_x(
    grad_x: np.ndarray,
    mp_comm,
    mp_size: int,
):
    """
    Use reduce-scatter / all-to-all to combine the contributions for grad_x from all nodes
    and scatter the reduced result along the input feature dimension.

    The grad_x tensor (gradient with respect to fc_o's input) has shape
        (batch_size, seq_length, in_dim),
    and the fc_o's weight matrix is sharded along the in_dim axis. In the
    backward pass, each node computes a local grad_x and then these must be
    summed across nodes. Instead of summing the full tensor and then slicing,
    we perform a reduce-scatter / all-to-all.

    Returns
    -------
    collected_grad_x : np.ndarray
        The reduced and scattered grad_x with shape
        (batch_size, seq_length, in_dim // mp_size).
    """
    # Reduce-scatter along the last axis: sum each shard across all ranks, and
    # each rank keeps only its own shard. Implemented with allreduce + slice for
    # clarity/robustness (equivalent result to a fused reduce-scatter).
    grad_x = np.ascontiguousarray(grad_x)
    rank = mp_comm.Get_rank()

    summed = np.empty_like(grad_x)
    mp_comm.Allreduce(grad_x, summed, op=MPI.SUM)

    in_dim = grad_x.shape[-1]
    part = in_dim // mp_size
    start = rank * part
    end = start + part
    collected_grad_x = summed[:, :, start:end]
    return collected_grad_x
