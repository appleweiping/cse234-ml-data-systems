import functools
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

import auto_diff as ad
import torch
from torchvision import datasets, transforms

max_len = 28


def transformer(X: ad.Node, nodes: List[ad.Node],
                      model_dim: int, seq_length: int, eps, batch_size, num_classes) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, input_dim), denoting the input data.
    nodes: List[ad.Node]
        Weight nodes for the transformer, in order:
        [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2].
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length
        for classification, in shape (batch_size, num_classes).
    """
    W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2 = nodes

    # --- Self-attention ---
    # X: (B, S, input_dim); W_*: (input_dim, model_dim) -> Q/K/V: (B, S, model_dim)
    Q = ad.matmul(X, W_Q)
    K = ad.matmul(X, W_K)
    V = ad.matmul(X, W_V)

    # Scaled dot-product attention.
    # scores = Q @ K^T / sqrt(model_dim) : (B, S, S)
    K_T = ad.transpose(K, -1, -2)
    scores = ad.matmul(Q, K_T) / (model_dim ** 0.5)
    attn = ad.softmax(scores, dim=-1)          # (B, S, S)
    context = ad.matmul(attn, V)               # (B, S, model_dim)

    # Output projection + residual + layernorm.
    attn_out = ad.matmul(context, W_O)         # (B, S, model_dim)
    x1 = attn_out + Q                          # residual on the query projection
    x1 = ad.layernorm(x1, normalized_shape=[model_dim], eps=eps)

    # --- Feed-forward network ---
    # W_1: (model_dim, model_dim), b_1: (model_dim,)
    h = ad.matmul(x1, W_1) + b_1               # (B, S, model_dim)
    h = ad.relu(h)
    # W_2: (model_dim, num_classes), b_2: (num_classes,)
    logits = ad.matmul(h, W_2) + b_2           # (B, S, num_classes)

    # Average over the sequence dimension -> (B, num_classes).
    output = ad.mean(logits, dim=(1,), keepdim=False)
    return output


def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node of shape (batch_size, num_classes), containing the logits.

    y_one_hot: ad.Node
        A node of shape (batch_size, num_classes), the one-hot ground truth.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax cross-entropy loss over the batch (scalar).
    """
    # probs = softmax(Z); cross entropy = -sum(y * log(probs)) / batch_size
    probs = ad.softmax(Z, dim=-1)
    log_probs = ad.log(probs)
    per_example = ad.sum_op(y_one_hot * log_probs, dim=(1,), keepdim=False)  # (B,)
    total = ad.sum_op(per_example, dim=(0,), keepdim=False)                  # scalar
    loss = ad.mul_by_const(total, -1.0 / batch_size)
    return loss


def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the transformer classifier."""
    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size
    total_loss = 0.0

    for i in range(num_batches):
        start_idx = i * batch_size
        if start_idx + batch_size > num_examples:
            continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]

        # Forward + backward in one evaluator run.
        # f_run_model returns [logits, loss, *grads].
        result = f_run_model(model_weights, X_batch, y_batch)
        logits, loss = result[0], result[1]
        grads = result[2:]

        # SGD update. Grads for 3-D weights carry a leading batch dim that must
        # be summed out before the update (each example contributes a gradient).
        for j in range(len(model_weights)):
            g = grads[j]
            while g.dim() > model_weights[j].dim():
                g = g.sum(dim=0)
            model_weights[j] = model_weights[j] - lr * g

        total_loss += float(loss) * (end_idx - start_idx)

    average_loss = total_loss / num_examples
    print('Avg_loss:', average_loss)
    return model_weights, average_loss


def train_model():
    """Train a single-layer transformer classifier on MNIST rows-as-sequence."""
    # Hyperparameters
    input_dim = 28          # Each row of the MNIST image
    seq_length = max_len    # Number of rows in the MNIST image
    num_classes = 10
    model_dim = 128
    eps = 1e-5

    # Training settings.
    num_epochs = 20
    batch_size = 50
    lr = 0.02

    # --- Build the forward graph ---
    X = ad.Variable(name="X")
    W_Q = ad.Variable(name="W_Q")
    W_K = ad.Variable(name="W_K")
    W_V = ad.Variable(name="W_V")
    W_O = ad.Variable(name="W_O")
    W_1 = ad.Variable(name="W_1")
    W_2 = ad.Variable(name="W_2")
    b_1 = ad.Variable(name="b_1")
    b_2 = ad.Variable(name="b_2")
    weight_nodes = [W_Q, W_K, W_V, W_O, W_1, W_2, b_1, b_2]

    y_predict: ad.Node = transformer(
        X, weight_nodes, model_dim, seq_length, eps, batch_size, num_classes
    )
    y_groundtruth = ad.Variable(name="y")
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)

    # --- Backward graph ---
    grads: List[ad.Node] = ad.gradients(loss, weight_nodes)

    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # --- Load the dataset (MNIST) ---
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    X_train = train_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().reshape(-1, 28, 28) / 255.0
    y_test = test_dataset.targets.numpy()

    encoder = OneHotEncoder(sparse_output=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    # --- Initialize model weights ---
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_K_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_V_val = np.random.uniform(-stdv, stdv, (input_dim, model_dim))
    W_O_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_1_val = np.random.uniform(-stdv, stdv, (model_dim, model_dim))
    W_2_val = np.random.uniform(-stdv, stdv, (model_dim, num_classes))
    b_1_val = np.random.uniform(-stdv, stdv, (model_dim,))
    b_2_val = np.random.uniform(-stdv, stdv, (num_classes,))

    def f_run_model(model_weights, X_batch, y_batch):
        """Compute forward + backward, returning [logits, loss, *grads]."""
        return evaluator.run(
            input_values={
                X: X_batch,
                y_groundtruth: y_batch,
                W_Q: model_weights[0],
                W_K: model_weights[1],
                W_V: model_weights[2],
                W_O: model_weights[3],
                W_1: model_weights[4],
                W_2: model_weights[5],
                b_1: model_weights[6],
                b_2: model_weights[7],
            }
        )

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """Compute only the forward graph and return predicted classes."""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size
        all_logits = []
        for i in range(num_batches):
            start_idx = i * batch_size
            if start_idx + batch_size > num_examples:
                continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            logits = test_evaluator.run({
                X: X_batch,
                W_Q: model_weights[0],
                W_K: model_weights[1],
                W_V: model_weights[2],
                W_O: model_weights[3],
                W_1: model_weights[4],
                W_2: model_weights[5],
                b_1: model_weights[6],
                b_2: model_weights[7],
            })
            all_logits.append(logits[0].detach().cpu().numpy())
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # --- Train ---
    X_train, X_test, y_train, y_test = (
        torch.tensor(X_train),
        torch.tensor(X_test),
        torch.DoubleTensor(y_train),
        torch.DoubleTensor(y_test),
    )
    model_weights: List[torch.Tensor] = [
        torch.tensor(W_Q_val), torch.tensor(W_K_val), torch.tensor(W_V_val),
        torch.tensor(W_O_val), torch.tensor(W_1_val), torch.tensor(W_2_val),
        torch.tensor(b_1_val), torch.tensor(b_2_val),
    ]

    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train, y_train)
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        predict_label = f_eval_model(X_test, model_weights)
        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label == y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    predict_label = f_eval_model(X_test, model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
