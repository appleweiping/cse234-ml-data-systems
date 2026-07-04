"""
CPU-runnable NumPy reference of the PA2 Part 1 tiled `D = ReLU(A @ B + C)` kernel.

The graded assignment implements this in Triton and is benchmarked on a Colab T4
GPU (fp16 tensor cores). This machine is CPU-only with no CUDA/Triton, so the
Triton kernel in `matmul_triton.ipynb` cannot run here. To still verify the
*algorithm* end-to-end, this module reproduces the exact same tiling structure
with NumPy:

  1. Tile assignment      -> loops over (pid_m, pid_n) output tiles
  2. Shared-memory tiling  -> loads BLOCK_M x BLOCK_K and BLOCK_K x BLOCK_N blocks
  3. Register accumulation -> per-tile fp32 accumulator
  4. Operator fusion       -> add C then ReLU, fused into the epilogue
  5. Epilogue / write cache-> masked store back into D (handles ragged tiles)

Run:  python matmul_tiled_cpu.py
"""

import time
import numpy as np


def matmul_add_relu_tiled(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
) -> np.ndarray:
    """Compute D = ReLU(A @ B + C) using an explicit tiled schedule.

    Mirrors the Triton kernel's tile assignment, cooperative K-loop fetching,
    register accumulation, and fused add+ReLU epilogue. Masks (via clamped
    slices) handle output/K tiles that do not divide evenly.
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Incompatible dimensions"
    assert C.shape == (M, N), "C must match output shape"

    D = np.empty((M, N), dtype=np.float32)

    num_pid_m = (M + block_m - 1) // block_m
    num_pid_n = (N + block_n - 1) // block_n
    num_pid_k = (K + block_k - 1) // block_k

    for pid_m in range(num_pid_m):
        m0 = pid_m * block_m
        m1 = min(m0 + block_m, M)            # epilogue mask on M
        for pid_n in range(num_pid_n):
            n0 = pid_n * block_n
            n1 = min(n0 + block_n, N)        # epilogue mask on N

            # Step 2/3: register accumulator for this output tile (fp32).
            acc = np.zeros((m1 - m0, n1 - n0), dtype=np.float32)

            # Step 3: cooperative fetching over K, block by block.
            for pid_k in range(num_pid_k):
                k0 = pid_k * block_k
                k1 = min(k0 + block_k, K)     # ragged-K mask
                a_tile = A[m0:m1, k0:k1]      # BLOCK_M x BLOCK_K
                b_tile = B[k0:k1, n0:n1]      # BLOCK_K x BLOCK_N
                acc += a_tile.astype(np.float32) @ b_tile.astype(np.float32)

            # Step 4: fuse add(C) then ReLU.
            acc += C[m0:m1, n0:n1].astype(np.float32)
            np.maximum(acc, 0.0, out=acc)

            # Step 5: epilogue store back into D.
            D[m0:m1, n0:n1] = acc

    return D


def reference_matmul_add_relu(A, B, C):
    return np.maximum(A.astype(np.float32) @ B.astype(np.float32) + C.astype(np.float32), 0.0)


def _run_checks():
    rng = np.random.default_rng(0)
    cases = [
        (128, 64, 128),
        (200, 130, 90),     # non-divisible dims exercise the masks
        (512, 256, 512),
        (1024, 512, 1024),
    ]
    print(f"{'M':>5} {'K':>5} {'N':>5} | {'max_abs_err':>12} | {'tiled_ms':>9} {'numpy_ms':>9}")
    print("-" * 60)
    all_ok = True
    for (M, K, N) in cases:
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        C = rng.standard_normal((M, N)).astype(np.float32)

        t0 = time.perf_counter()
        D = matmul_add_relu_tiled(A, B, C, block_m=64, block_n=64, block_k=32)
        t_tiled = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        R = reference_matmul_add_relu(A, B, C)
        t_np = (time.perf_counter() - t0) * 1e3

        err = float(np.max(np.abs(D - R)))
        ok = np.allclose(D, R, atol=1e-3, rtol=1e-3)
        all_ok = all_ok and ok
        flag = "OK" if ok else "FAIL"
        print(f"{M:>5} {K:>5} {N:>5} | {err:>12.2e} | {t_tiled:>9.2f} {t_np:>9.2f}  {flag}")

    print("-" * 60)
    print("ALL TILED == REFERENCE:", all_ok)
    return all_ok


if __name__ == "__main__":
    ok = _run_checks()
    raise SystemExit(0 if ok else 1)
