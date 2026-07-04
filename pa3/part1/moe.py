import numpy as np

from mpiwrapper import mpi
from rng import get_rng, rng_context


class Linear:
    """Simple linear layer y = xW + b"""

    def __init__(self, in_features, out_features):
        # Use default RNG for all other operations - no need for context
        self.weight = get_rng().randn(in_features, out_features) * 0.01
        self.bias = np.zeros(out_features)

    def __call__(self, x):
        return np.dot(x, self.weight) + self.bias


class Expert:
    """Expert network with one hidden layer and ReLU activation"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        # Use rank-specific RNG for expert initialization
        with rng_context('expert'):
            self.fc1 = Linear(input_dim, hidden_dim)
            self.fc2 = Linear(hidden_dim, output_dim)

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)  # ReLU
        return self.fc2(hidden)


class Router:
    """Routes inputs to experts using softmax-based gating"""

    def __init__(self, input_dim, num_experts):
        # Router should be consistent across all ranks, so use default RNG
        self.linear = Linear(input_dim, num_experts)

    def __call__(self, x, topk=1):
        logits = self.linear(x)

        # Softmax for routing probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Select top-k experts
        indices = np.argsort(-probs, axis=1)[:, :topk]
        gates = np.take_along_axis(probs, indices, axis=1)

        # Normalize gates to sum to 1
        gates = gates / np.sum(gates, axis=1, keepdims=True)

        return indices, gates


class ShardedLinear:
    """
    Linear layer that is sharded across processes
    Each process only holds a portion of the weight matrix
    
    Requires that out_features is evenly divisible by the world size
    """

    def __init__(self, in_features, out_features):
        self.rank = mpi.get_rank()
        self.world_size = mpi.get_size()

        # Assert that out_features is evenly divisible by world_size
        assert out_features % self.world_size == 0, f"Output features ({out_features}) must be evenly divisible by world size ({self.world_size})"

        # Calculate the local output dimension
        self.out_features_global = out_features
        self.local_out_features = out_features // self.world_size

        # Calculate output offset for this rank (simple with even division)
        self.output_offset = self.rank * self.local_out_features

        # Initialize local weights and bias
        self.weight = get_rng().randn(in_features, self.local_out_features) * 0.01
        self.bias = get_rng().randn(self.local_out_features)

    def __call__(self, x):
        # Handle empty batch case
        if x.shape[0] == 0:
            return np.zeros((0, self.out_features_global))

        # Each rank owns the output columns [output_offset : output_offset+local].
        # Compute the local partial output, then all-gather the shards across all
        # ranks and stitch them back together along the output dimension.
        local_out = np.dot(x, self.weight) + self.bias   # (batch, local_out_features)
        local_out = np.ascontiguousarray(local_out.astype(np.float32))

        gathered = mpi.allgather(local_out)  # list of world_size arrays, index = rank
        result = np.concatenate(gathered, axis=1).astype(np.float32)
        return result


class ShardedExpert:
    """Expert network with one hidden layer and ReLU activation, sharded across processes"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        # Use rank-specific RNG for expert initialization
        with rng_context('expert'):
            self.fc1 = ShardedLinear(input_dim, hidden_dim)
            self.fc2 = ShardedLinear(hidden_dim, output_dim)

    def __call__(self, x):
        hidden = self.fc1(x)
        hidden = np.maximum(0, hidden)  # ReLU
        return self.fc2(hidden)


class MoE_TP:
    """
    Distributed Mixture of Experts using MPI for tensor parallelism
    
    TP-style MoE:
    - Each process holds a portion of every expert (sharded experts)
    - Router is replicated on all processes
    - All-to-all and all-gather communication patterns for processing
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for each expert
        output_dim (int): Output dimension
        num_experts (int): Total number of experts in the model
        topk (int): Number of experts to route each input to
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)
        self.rank = mpi.get_rank()
        self.world_size = mpi.get_size()

        # Create router (replicated on all processes)
        with rng_context('router'):
            self.router = Router(input_dim, num_experts)

        # Create sharded experts - each expert is sharded across all processes
        with rng_context('expert'):
            self.experts = [ShardedExpert(input_dim, hidden_dim, output_dim)
                            for _ in range(num_experts)]

        print(f"[Rank {self.rank}] Holding portions of all {num_experts} experts")

    def forward(self, x):
        """
        Distributed forward pass through the MoE model using tensor parallelism
        with optimized batch processing
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Initialize output tensor
        outputs = np.zeros((batch_size, self.output_dim))

        # TP-style MoE: every rank holds a shard of *every* expert. Each
        # ShardedExpert call internally all-gathers its output shards, so it
        # returns the full expert output on all ranks. The router is replicated,
        # so all ranks compute the same routing and the same combined output.
        # 1. Routing.
        indices, gates = self.router(x, self.topk)
        # 2. Combine the (sharded) expert outputs, weighted by the gates.
        for k in range(self.topk):
            for i in range(batch_size):
                expert_idx = indices[i, k]
                gate = gates[i, k]
                item = x[i:i + 1]                       # (1, input_dim)
                expert_output = self.experts[expert_idx](item)  # TP all-gather inside
                outputs[i] += gate * expert_output[0]

        return outputs

    def __call__(self, x):
        return self.forward(x)


class SimpleMoE:
    """
    Simple reference implementation of Mixture of Experts.
    
    This class implements a basic MoE model that routes inputs to a subset
    of experts and combines their outputs using learned gating weights.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for each expert
        output_dim (int): Output dimension
        num_experts (int): Number of expert networks
        topk (int): Number of experts to route each input to
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.topk = min(topk, num_experts)

        # Create router network
        with rng_context('router'):
            self.router = Router(input_dim, num_experts)

        # Create expert networks
        with rng_context('expert'):
            self.experts = [Expert(input_dim, hidden_dim, output_dim)
                            for _ in range(num_experts)]

    def forward(self, x):
        """
        Forward pass through the MoE model
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Get expert assignments and gates
        indices, gates = self.router(x, self.topk)

        # Initialize output tensor
        outputs = np.zeros((batch_size, self.output_dim))

        # Compute weighted combination of expert outputs
        for k in range(self.topk):
            for i in range(batch_size):
                expert_idx = indices[i, k]
                gate = gates[i, k]
                item = x[i:i + 1]  # (1, input_dim)
                expert_output = self.experts[expert_idx](item)
                outputs[i] += gate * expert_output[0]

        return outputs

    def __call__(self, x):
        return self.forward(x)


class MoE_EP:
    """
    Distributed Mixture of Experts using MPI for expert parallelism
    
    EP-style MoE: 
    Each process hosts exactly one expert. Router is replicated on all processes.
    
    Args:
        input_dim (int): Input feature dimension
        hidden_dim (int): Hidden dimension for each expert
        output_dim (int): Output dimension
        topk (int): Number of experts to route each input to
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, topk=1):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts  # Total number of processes = number of experts
        self.topk = min(topk, self.num_experts)
        self.rank = mpi.get_rank()

        # Create router (replicated on all processes)
        with rng_context('router'):
            self.router = Router(input_dim, self.num_experts)

        # Create only one expert per process
        with rng_context('expert_with_rank'):
            self.expert = Expert(input_dim, hidden_dim, output_dim)

    def forward(self, x):
        """
        Distributed forward pass through the MoE model
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Initialize output tensor
        outputs = np.zeros((batch_size, self.output_dim))

        # EP-style MoE: rank r hosts expert r in its entirety. The router is
        # replicated so every rank knows the full routing table.
        # 1. Routing (identical on all ranks).
        indices, gates = self.router(x, self.topk)

        # 2. Each rank runs ITS expert over only the tokens routed to it. We
        #    build a boolean mask of which tokens select this rank's expert
        #    (in any of their top-k slots), run the expert on that gathered
        #    slice, and scatter the results back to a full-batch buffer.
        my_expert = self.rank
        token_mask = np.any(indices == my_expert, axis=1)  # (batch,)
        local_full = np.zeros((batch_size, self.output_dim), dtype=np.float64)
        if np.any(token_mask):
            sel = np.where(token_mask)[0]
            local_out = self.expert(x[sel])               # (num_sel, output_dim)
            local_full[sel] = local_out

        # 3. All-to-all / all-gather each expert's full-batch contribution so
        #    every rank can assemble the final combination. (allgather of the
        #    per-expert full-batch buffers is the communication step.)
        all_expert_out = mpi.allgather(local_full)  # list indexed by expert/rank

        # 4. Combine with the gates: for each token and each of its top-k picks,
        #    add gate * (that expert's output for this token).
        for i in range(batch_size):
            for k in range(self.topk):
                e = indices[i, k]
                gate = gates[i, k]
                outputs[i] += gate * all_expert_out[e][i]

        return outputs

    def __call__(self, x):
        return self.forward(x)
