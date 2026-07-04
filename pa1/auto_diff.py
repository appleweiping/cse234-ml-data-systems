from typing import Any, Dict, List

import torch


class Node:
    """Node in a computational graph.

    Fields
    ------
    inputs: List[Node]
        The list of input nodes to this node.

    op: Op
        The op of this node.

    attrs: Dict[str, Any]
        The attribute dictionary of this node.
        E.g. "constant" is the constant operand of add_by_const.

    name: str
        Name of the node for debugging purposes.
    """

    inputs: List["Node"]
    op: "Op"
    attrs: Dict[str, Any]
    name: str

    def __init__(
        self, inputs: List["Node"], op: "Op", attrs: Dict[str, Any] = {}, name: str = ""
    ) -> None:
        self.inputs = inputs
        self.op = op
        self.attrs = attrs
        self.name = name

    def __add__(self, other):
        if isinstance(other, Node):
            return add(self, other)
        else:
            assert isinstance(other, (int, float))
            return add_by_const(self, other)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            assert isinstance(other, (int, float))
            return mul_by_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            assert isinstance(other, (int, float))
            return div_by_const(self, other)

    # Allow left-hand-side add and multiplication.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow printing the node name."""
        return self.name

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        raise KeyError(f"Attribute {attr_name} does not exist in node {self}")

    __repr__ = __str__


class Variable(Node):
    """A variable node with given name."""

    def __init__(self, name: str) -> None:
        super().__init__(inputs=[], op=placeholder, name=name)


class Op:
    """The class of operations performed on nodes."""

    def __call__(self, *kwargs) -> Node:
        """Create a new node with this current op.

        Returns
        -------
        The created new node.
        """
        raise NotImplementedError

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Compute the output value of the given node with its input
        node values given.

        Parameters
        ----------
        node: Node
            The node whose value is to be computed

        input_values: List[torch.Tensor]
            The input values of the given node.

        Returns
        -------
        output: torch.Tensor
            The computed output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given a node and its output gradient node, compute partial
        adjoints with regards to each input node.

        Parameters
        ----------
        node: Node
            The node whose inputs' partial adjoints are to be computed.

        output_grad: Node
            The output gradient with regard to given node.

        Returns
        -------
        input_grads: List[Node]
            The list of partial gradients with regard to each input of the node.
        """
        raise NotImplementedError


class PlaceholderOp(Op):
    """The placeholder op to denote computational graph input nodes."""

    def __call__(self, name: str) -> Node:
        return Node(inputs=[], op=self, name=name)

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        raise RuntimeError(
            "Placeholder nodes have no inputs, and there values cannot be computed."
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        raise RuntimeError("Placeholder nodes have no inputs.")


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}+{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of input values."""
        assert len(input_values) == 2
        return input_values[0] + input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to each input."""
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}+{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] + node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to the input."""
        return [output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}*{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of input values."""
        assert len(input_values) == 2
        return input_values[0] * input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to each input."""
        return [output_grad * node.inputs[1], output_grad * node.inputs[0]]


class MulByConstOp(Op):
    """Op to element-wise multiply a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}*{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] * node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to the input."""
        return [output_grad * node.constant]

class GreaterThanOp(Op):
    """Op to compare if node_A > node_B element-wise."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}>{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return element-wise comparison result as float tensor."""
        assert len(input_values) == 2
        return (input_values[0] > input_values[1]).float()

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Comparison operations have gradient of 0."""
        return [zeros_like(node.inputs[0]), zeros_like(node.inputs[1])]

class SubOp(Op):
    """Op to element-wise subtract two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}-{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise subtraction of input values."""
        assert len(input_values) == 2
        return input_values[0] - input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of subtraction node, return partial adjoint to each input."""
        return [output_grad, mul_by_const(output_grad, -1)]

class ZerosLikeOp(Op):
    """Zeros-like op that returns an all-zero array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"ZerosLike({node_A.name})")

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return an all-zero tensor with the same shape as input."""
        assert len(input_values) == 1
        return torch.zeros_like(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]

class OnesLikeOp(Op):
    """Ones-like op that returns an all-one array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"OnesLike({node_A.name})")

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return an all-one tensor with the same shape as input."""
        assert len(input_values) == 1
        return torch.ones_like(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]

class SumOp(Op):
    """
    Op to compute sum along specified dimensions.

    Note: This is a reference implementation for SumOp.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, dim: tuple, keepdim: bool = False) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Sum({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return input_values[0].sum(dim=node.dim, keepdim=node.keepdim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Gradient of a sum reduction: broadcast output_grad back to the input
        shape. Rank-agnostic (handles keepdim True/False and any number of
        reduced dims), unlike the original 3-D-only reference which assumed
        expand_as_3d.
        """
        dim = node.attrs['dim']
        keepdim = node.attrs["keepdim"]
        dims = (dim,) if isinstance(dim, int) else tuple(dim)
        return [_SumGradOp()(node.inputs[0], output_grad, dims, keepdim)]

class ExpandAsOp(Op):
    """Op to broadcast a tensor to the shape of another tensor.

    Note: This is a reference implementation for ExpandAsOp.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast({node_A.name} -> {node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values
        return input_tensor.expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the broadcast node, compute partial adjoint to input."""

        return [sum_op(output_grad,dim=0), zeros_like(output_grad)]

class ExpandAsOp3d(Op):
    """Op to broadcast a tensor to the shape of another tensor.

    Note: This is a reference implementation for ExpandAsOp3d.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast({node_A.name} -> {node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values
        return input_tensor.unsqueeze(1).expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the broadcast node, compute partial adjoint to input."""

        return [sum_op(output_grad,dim=(0, 1)), zeros_like(output_grad)]

class LogOp(Op):
    """Logarithm (natural log) operation."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"Log({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the natural logarithm of the input."""
        assert len(input_values) == 1, "Log operation requires one input."
        return torch.log(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the Log node, return the partial adjoint to the input."""
        input_node = node.inputs[0]
        return [output_grad / input_node]


class BroadcastOp(Op):
    def __call__(self, node_A: Node, input_shape: List[int], target_shape: List[int]) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"input_shape": input_shape, "target_shape": target_shape},
            name=f"Broadcast({node_A.name}, {target_shape})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 1
        return input_values[0].expand(node.attrs["target_shape"])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of broadcast node, return partial adjoint to input.

        For broadcasting, we need to sum out the broadcasted dimensions to get
        back to the original shape.
        """
        if "input_shape" not in node.attrs:
            raise ValueError("Input shape is not set. Make sure compute() is called before gradient()")

        input_shape = node.attrs["input_shape"]
        output_shape = node.attrs["target_shape"]

        dims_to_sum = []
        for i, (in_size, out_size) in enumerate(zip(input_shape[::-1], output_shape[::-1])):
            if in_size != out_size:
                dims_to_sum.append(len(output_shape) - 1 - i)

        grad = output_grad
        if dims_to_sum:
            grad = sum_op(grad, dim=dims_to_sum, keepdim=True)

        if len(output_shape) > len(input_shape):
            grad = sum_op(grad, dim=list(range(len(output_shape) - len(input_shape))), keepdim=False)

        return [grad]

class DivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}/{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of input values."""
        assert len(input_values) == 2
        return input_values[0] / input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to each input.

        y = a / b
        dL/da = output_grad / b
        dL/db = -output_grad * a / b^2
        """
        a, b = node.inputs[0], node.inputs[1]
        grad_a = output_grad / b
        grad_b = mul_by_const(output_grad * a / (b * b), -1)
        return [grad_a, grad_b]

class DivByConstOp(Op):
    """Op to element-wise divide a nodes by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}/{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] / node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to the input."""
        return [output_grad / node.constant]

class TransposeOp(Op):
    """Op to transpose a matrix."""

    def __call__(self, node_A: Node, dim0: int, dim1: int) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim0": dim0, "dim1": dim1},
            name=f"transpose({node_A.name}, {dim0}, {dim1})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the transpose of the input by swapping two dimensions.

        For example:
        - transpose(x, 1, 0) swaps first two dimensions
        """
        assert len(input_values) == 1
        return torch.transpose(input_values[0], node.dim0, node.dim1)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of transpose node, return partial adjoint to input.

        Transpose is its own inverse: transpose the incoming gradient back
        along the same two dimensions.
        """
        return [transpose(output_grad, node.dim0, node.dim1)]

class MatMulOp(Op):
    """Matrix multiplication op of two nodes."""

    def __call__(
        self, node_A: Node, node_B: Node
    ) -> Node:
        """Create a matrix multiplication node.

        Parameters
        ----------
        node_A: Node
            The lhs matrix.
        node_B: Node
            The rhs matrix

        Returns
        -------
        result: Node
            The node of the matrix multiplication.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the matrix multiplication result of input values."""
        assert len(input_values) == 2
        return torch.matmul(input_values[0], input_values[1])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of matmul node, return partial adjoint to each input.

        For Y = A @ B (last two dims are the matrix dims):
            dL/dA = output_grad @ B^T
            dL/dB = A^T @ output_grad
        where the transpose swaps the last two dimensions so batched matmul works.
        """
        A, B = node.inputs[0], node.inputs[1]
        grad_A = matmul(output_grad, transpose(B, -1, -2))
        grad_B = matmul(transpose(A, -1, -2), output_grad)
        return [grad_A, grad_B]


class SoftmaxOp(Op):
    """Softmax operation on input node."""

    def __call__(self, node_A: Node, dim: int = -1) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim},
            name=f"Softmax({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return softmax of input along specified dimension."""
        assert len(input_values) == 1
        return torch.softmax(input_values[0], dim=node.dim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of softmax node, return partial adjoint to input.

        With s = softmax(x) along `dim`, the Jacobian-vector product is
            dL/dx = s * (g - sum(g * s, dim, keepdim=True))
        where g = output_grad.
        """
        dim = node.attrs["dim"]
        s = softmax(node.inputs[0], dim=dim)
        weighted = sum_op(output_grad * s, dim=(dim,), keepdim=True)
        return [s * (output_grad - weighted)]


class LayerNormOp(Op):
    """Layer normalization operation."""

    def __call__(self, node_A: Node, normalized_shape: List[int], eps: float = 1e-5) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"normalized_shape": normalized_shape, "eps": eps},
            name=f"LayerNorm({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return layer normalized input (no affine weight/bias)."""
        assert len(input_values) == 1
        x = input_values[0]
        normalized_shape = node.attrs["normalized_shape"]
        eps = node.attrs["eps"]
        dims = tuple(range(x.dim() - len(normalized_shape), x.dim()))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        return (x - mean) / torch.sqrt(var + eps)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """
        Given gradient of the LayerNorm node wrt its output, return partial
        adjoint (gradient) wrt the input x.

        For y = (x - mu) / sqrt(var + eps) over the last `len(normalized_shape)`
        dims of size N, the input gradient is
            dL/dx = (1 / (N * std)) * ( N*g - sum(g) - x_hat * sum(g * x_hat) )
        where x_hat = (x - mu)/std, std = sqrt(var+eps), g = output_grad, and the
        sums are over the normalized dimensions (keepdim=True).
        """
        x = node.inputs[0]
        normalized_shape = node.attrs["normalized_shape"]
        eps = node.attrs["eps"]
        N = 1
        for s in normalized_shape:
            N *= s
        # dims to reduce over are the last len(normalized_shape) dims.
        dim = tuple(range(-len(normalized_shape), 0))

        mean = mean_op(x, dim=dim, keepdim=True)
        x_mu = x - mean
        var = mean_op(power(x_mu, 2), dim=dim, keepdim=True)
        std = sqrt(var + eps)
        x_hat = x_mu / std

        g = output_grad
        sum_g = sum_op(g, dim=dim, keepdim=True)
        sum_g_xhat = sum_op(g * x_hat, dim=dim, keepdim=True)

        grad_x = (g * N - sum_g - x_hat * sum_g_xhat) / (std * N)
        return [grad_x]

class ReLUOp(Op):
    """ReLU activation function."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"ReLU({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return ReLU of input."""
        assert len(input_values) == 1
        return torch.relu(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of ReLU node, return partial adjoint to input.

        The mask is 1 where x > 0 and 0 elsewhere; expressed with graph ops as
        greater(x, 0) which produces a float 0/1 tensor.
        """
        x = node.inputs[0]
        mask = greater(x, zeros_like(x))
        return [output_grad * mask]

class SqrtOp(Op):
    """Op to compute element-wise square root."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"Sqrt({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.sqrt(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """d/dx sqrt(x) = 1 / (2 sqrt(x))."""
        x = node.inputs[0]
        return [output_grad / (sqrt(x) * 2)]

class PowerOp(Op):
    """Op to compute element-wise power."""

    def __call__(self, node_A: Node, exponent: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"exponent": exponent},
            name=f"Power({node_A.name}, {exponent})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.pow(input_values[0], node.exponent)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """d/dx x^p = p * x^(p-1)."""
        x = node.inputs[0]
        p = node.attrs["exponent"]
        return [output_grad * (power(x, p - 1) * p)]

class MeanOp(Op):
    """Op to compute mean along specified dimensions."""

    def __call__(self, node_A: Node, dim: tuple, keepdim: bool = False) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Mean({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return input_values[0].mean(dim=node.dim, keepdim=node.keepdim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Mean divides by the number of reduced elements, then broadcasts the
        gradient back to the input shape. Works for any rank and either keepdim."""
        x = node.inputs[0]
        dim = node.attrs["dim"]
        keepdim = node.attrs["keepdim"]
        dims = (dim,) if isinstance(dim, int) else tuple(dim)
        return [_mean_grad(x, output_grad, dims, keepdim)]

# Create global instances of ops.
# Your implementation should just use these instances, rather than creating new instances.
placeholder = PlaceholderOp()
add = AddOp()
mul = MulOp()
div = DivOp()
add_by_const = AddByConstOp()
mul_by_const = MulByConstOp()
div_by_const = DivByConstOp()
matmul = MatMulOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
softmax = SoftmaxOp()
layernorm = LayerNormOp()
relu = ReLUOp()
transpose = TransposeOp()
mean = MeanOp()
mean_op = mean
sum_op = SumOp()
sqrt = SqrtOp()
power = PowerOp()
greater = GreaterThanOp()
expand_as = ExpandAsOp()
expand_as_3d = ExpandAsOp3d()
log = LogOp()
sub = SubOp()
broadcast = BroadcastOp()


def _mean_grad(x_node: "Node", output_grad: "Node", dims: tuple, keepdim: bool) -> "Node":
    """Helper to build the gradient of a mean reduction.

    grad_input = broadcast(output_grad / count) to x's shape.
    Implemented at graph level via a small custom op so it works for any rank
    and both keepdim settings.
    """
    return _MeanGradOp()(x_node, output_grad, dims, keepdim)


class _MeanGradOp(Op):
    """Internal op producing the input gradient of a mean reduction."""

    def __call__(self, x_node: Node, grad_node: Node, dims: tuple, keepdim: bool) -> Node:
        return Node(
            inputs=[x_node, grad_node],
            op=self,
            attrs={"dims": dims, "keepdim": keepdim},
            name=f"MeanGrad({x_node.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        x, g = input_values
        dims = node.attrs["dims"]
        keepdim = node.attrs["keepdim"]
        ndims = tuple(d % x.dim() for d in dims)
        count = 1
        for d in ndims:
            count *= x.shape[d]
        if not keepdim:
            for d in sorted(ndims):
                g = g.unsqueeze(d)
        return (g / count).expand_as(x)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        # Not needed for second-order through mean in this assignment.
        return [zeros_like(node.inputs[0]), zeros_like(node.inputs[1])]


class _SumGradOp(Op):
    """Internal op producing the input gradient of a sum reduction.

    Broadcasts the (possibly reduced) output gradient back to the input's shape.
    Works for any rank and either keepdim setting.
    """

    def __call__(self, x_node: Node, grad_node: Node, dims: tuple, keepdim: bool) -> Node:
        return Node(
            inputs=[x_node, grad_node],
            op=self,
            attrs={"dims": dims, "keepdim": keepdim},
            name=f"SumGrad({x_node.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        x, g = input_values
        dims = node.attrs["dims"]
        keepdim = node.attrs["keepdim"]
        ndims = tuple(d % x.dim() for d in dims)
        if not keepdim:
            for d in sorted(ndims):
                g = g.unsqueeze(d)
        return g.expand_as(x).clone()

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        # Second-order through sum reduces back over the same dims.
        dims = node.attrs["dims"]
        return [zeros_like(node.inputs[0]), sum_op(output_grad, dim=dims, keepdim=node.attrs["keepdim"])]


def topological_sort(nodes):
    """Perform a topological sort on the computational graph.

    Parameters
    ----------
    nodes : List[Node] or Node
        Node(s) whose dependencies should all appear before them in the output.

    Returns
    -------
    List[Node]
        Nodes in topological order (inputs before the nodes that use them).
    """
    if isinstance(nodes, Node):
        nodes = [nodes]

    visited = set()
    order: List[Node] = []

    def dfs(n: Node):
        if id(n) in visited:
            return
        visited.add(id(n))
        for inp in n.inputs:
            dfs(inp)
        order.append(n)

    for n in nodes:
        dfs(n)
    return order

class Evaluator:
    """The node evaluator that computes the values of nodes in a computational graph."""

    eval_nodes: List[Node]

    def __init__(self, eval_nodes: List[Node]) -> None:
        """Constructor, which takes the list of nodes to evaluate in the computational graph.

        Parameters
        ----------
        eval_nodes: List[Node]
            The list of nodes whose values are to be computed.
        """
        self.eval_nodes = eval_nodes

    def run(self, input_values: Dict[Node, torch.Tensor]) -> List[torch.Tensor]:
        """Computes values of nodes in `eval_nodes` field with
        the computational graph input values given by the `input_values` dict.

        Parameters
        ----------
        input_values: Dict[Node, torch.Tensor]
            The dictionary providing the values for input nodes of the
            computational graph.
            Throw ValueError when the value of any needed input node is
            not given in the dictionary.

        Returns
        -------
        eval_values: List[torch.Tensor]
            The list of values for nodes in `eval_nodes` field.
        """
        # id-keyed cache so distinct Node objects with the same name don't collide.
        node_values: Dict[int, torch.Tensor] = {}
        for node, val in input_values.items():
            node_values[id(node)] = val

        for node in topological_sort(self.eval_nodes):
            if id(node) in node_values:
                continue
            if isinstance(node.op, PlaceholderOp):
                raise ValueError(
                    f"Value of placeholder/variable node '{node.name}' was not provided."
                )
            inputs = [node_values[id(inp)] for inp in node.inputs]
            node_values[id(node)] = node.op.compute(node, inputs)

        return [node_values[id(node)] for node in self.eval_nodes]


def gradients(output_node: Node, nodes: List[Node]) -> List[Node]:
    """Construct the backward computational graph, which takes gradient
    of given output node with respect to each node in input list.
    Return the list of gradient nodes, one for each node in the input list.

    Parameters
    ----------
    output_node: Node
        The output node to take gradient of, whose gradient is 1.

    nodes: List[Node]
        The list of nodes to take gradient with regard to.

    Returns
    -------
    grad_nodes: List[Node]
        A list of gradient nodes, one for each input nodes respectively.
    """
    # Reverse-mode autodiff: accumulate partial adjoints per node, then combine.
    node_to_grads: Dict[int, List[Node]] = {id(output_node): [ones_like(output_node)]}

    reverse_topo = reversed(topological_sort([output_node]))
    node_to_output_grad: Dict[int, Node] = {}

    for node in reverse_topo:
        grads_list = node_to_grads.get(id(node))
        if grads_list is None:
            continue
        # Sum all incoming adjoints for this node.
        grad = grads_list[0]
        for extra in grads_list[1:]:
            grad = grad + extra
        node_to_output_grad[id(node)] = grad

        if len(node.inputs) == 0:
            continue

        input_grads = node.op.gradient(node, grad)
        for inp, inp_grad in zip(node.inputs, input_grads):
            node_to_grads.setdefault(id(inp), []).append(inp_grad)

    return [node_to_output_grad[id(n)] for n in nodes]
