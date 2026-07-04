from typing import Any, Dict, List
import torch
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation.

    Computes LayerNorm(A @ B) in a single op. The forward pass fuses the two
    kernels; the backward pass reuses the analytic MatMul and LayerNorm adjoints
    but chains them together so the intermediate (A @ B) never needs a separate
    graph node.
    """

    def __call__(
        self,
        node_A: Node,
        node_B: Node,
        normalized_shape: List[int],
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        a, b = input_values
        y = torch.matmul(a, b)
        normalized_shape = node.attrs["normalized_shape"]
        eps = node.attrs["eps"]
        dims = tuple(range(y.dim() - len(normalized_shape), y.dim()))
        mean = y.mean(dim=dims, keepdim=True)
        var = y.var(dim=dims, unbiased=False, keepdim=True)
        return (y - mean) / torch.sqrt(var + eps)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input.

        Let Z = A @ B and Y = LayerNorm(Z). We reconstruct Z as a graph node,
        route output_grad through the LayerNorm adjoint to get dL/dZ, then apply
        the MatMul adjoints:
            dL/dA = dL/dZ @ B^T
            dL/dB = A^T @ dL/dZ
        """
        a, b = node.inputs[0], node.inputs[1]
        normalized_shape = node.attrs["normalized_shape"]
        eps = node.attrs["eps"]

        z = matmul(a, b)
        ln = layernorm(z, normalized_shape=normalized_shape, eps=eps)
        grad_z = ln.op.gradient(ln, output_grad)[0]

        grad_a = matmul(grad_z, transpose(b, -1, -2))
        grad_b = matmul(transpose(a, -1, -2), grad_z)
        return [grad_a, grad_b]


class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation.

    Computes Softmax(A @ B, dim) in a single op, with a fused forward and a
    chained backward through the analytic MatMul and Softmax adjoints.
    """

    def __call__(
        self,
        node_A: Node,
        node_B: Node,
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        a, b = input_values
        y = torch.matmul(a, b)
        return torch.softmax(y, dim=node.attrs["dim"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input.

        Let Z = A @ B and Y = Softmax(Z). We reconstruct Z, route output_grad
        through the Softmax adjoint to get dL/dZ, then apply the MatMul adjoints.
        """
        a, b = node.inputs[0], node.inputs[1]
        dim = node.attrs["dim"]

        z = matmul(a, b)
        sm = softmax(z, dim=dim)
        grad_z = sm.op.gradient(sm, output_grad)[0]

        grad_a = matmul(grad_z, transpose(b, -1, -2))
        grad_b = matmul(transpose(a, -1, -2), grad_z)
        return [grad_a, grad_b]

# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()
