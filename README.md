# CSE 234 — Data Systems for Machine Learning (Programming Assignments)

> From-skeleton implementations of the three programming assignments of
> **CSE 234: Data Systems for Machine Learning** (UC San Diego, Winter 2025, Prof. Hao Zhang),
> part of a [csdiy.wiki](https://csdiy.wiki/) full-catalog build. Builds an autodiff engine +
> transformer, MPI collective-communication / tensor-&-data parallelism, a Mixture-of-Experts
> layer with TP/EP, scaling-law cost analysis, and speculative decoding — all real, all run on CPU.

![status](https://img.shields.io/badge/status-complete-brightgreen)
![language](https://img.shields.io/badge/Python-informational)
![license](https://img.shields.io/badge/license-MIT-blue)

## Overview

CSE 234 is a graduate ML-systems course covering how modern deep-learning systems are built and
scaled: automatic differentiation, operator fusion, tensor/data/expert parallelism, collective
communication, scaling laws, and inference acceleration. This repo implements the three
programming assignments (PA1–PA3) starting from the official
[`hao-ai-lab/cse234-w25-PA`](https://github.com/hao-ai-lab/cse234-w25) skeleton and filling in every
computational core:

- **PA1** — a reverse-mode automatic-differentiation engine (à la a mini-PyTorch autograd), fused
  operators (`MatMul+LayerNorm`, `MatMul+Softmax`), and a transformer classifier trained on MNIST.
- **PA2** — collective communication primitives (`myAllreduce`, `myAlltoall`) from scratch over
  MPI, plus data-parallel data splitting and naive tensor-model-parallel forward/backward comms
  for a transformer layer, and a CPU tiled matmul kernel.
- **PA3** — a Mixture-of-Experts layer with **tensor-parallel** and **expert-parallel** variants,
  transformer training-cost / scaling-law analysis (Llama-7B, DeepSeek-V3), and **speculative
  decoding** with a Pythia draft/target model pair.

Everything here runs **CPU-only** on Windows (`OMP_NUM_THREADS=3`); the numbers below are measured
on this machine, not claimed.

## Results (measured on CPU, Windows, `OMP_NUM_THREADS=3`, MS-MPI 10.1, Python 3.11)

| Assignment | What it does | Result (measured) |
|---|---|---|
| **PA1** autodiff + fused ops | Reverse-mode autodiff engine, all ops + fused kernels | **40/40 pytest passed** (`results/pa1/pytest_results.txt`) |
| **PA1** transformer on MNIST | Single-layer transformer classifier, 20 epochs | **final test accuracy 0.8537**, loss 0.463 (`results/pa1/train_mnist.log`) |
| **PA2** collective comms | Custom `myAllreduce` / `myAlltoall` vs MPI, 8 ranks, 100 runs | all correct; myAllreduce 120 µs (MPI 50 µs), myAlltoall 15.8 ms (MPI 7.9 ms) |
| **PA2** parallel-training comms | data-split + get_info + TP forward/backward pytest | **10/10 passed** across 4 suites (`results/pa2/pytest_results.txt`) |
| **PA2** tiled matmul (CPU) | tiled GEMM vs NumPy, correctness | max abs err ≤ 6.9e-5, **ALL == REFERENCE** |
| **PA3.1** MoE (TP/EP) | Simple / Tensor-Parallel / Expert-Parallel MoE, 5 ranks | all 3 correctness tests **passed**; EP ~**10–16× faster than TP** |
| **PA3.2** cost analysis | Llama-7B & DeepSeek-V3 params/FLOPs/memory, scaling-law optimizer | Llama-7B: **6.738B params**, 0.898 TFLOPs/layer, 0.386 GB; optimal GPU = **A100** |
| **PA3.3** speculative decoding | pythia-410m target + pythia-70m draft, k=8, CPU | output **== baseline** (lossless); 1.16× speedup / 53% acceptance on prompt 1 |

Sample generated output is byte-for-byte identical to greedy baseline decoding (`output==baseline: True`),
confirming the speculative decoder is correctness-preserving (`results/pa3/speculative_decode.log`).

### MoE TP vs EP forward-pass time (measured, `feature_dim=32, hidden=128, out=64, top-k=2, 4 experts`)

```
 batch |   simple ms |       TP ms |       EP ms |   EP/TP
------------------------------------------------------------
     8 |       0.245 |       5.032 |       0.525 |     0.10
    32 |       1.854 |      13.648 |       0.946 |     0.07
   128 |       6.117 |      50.819 |       3.437 |     0.07
   512 |      23.815 |     205.402 |      12.853 |     0.06
```

Expert parallelism replaces the many tiny per-token all-gathers of the naive TP-MoE with a single
large collective per layer — see `pa3/part1/analysis.md` for the full write-up.

## Implemented assignments

- [x] **PA1.1 Automatic differentiation** — `Node`/`Op` graph, forward eval + topological reverse-mode backward for add/mul/div/matmul (2D & 3D)/transpose/broadcast/sqrt/power/relu/softmax/layernorm.
- [x] **PA1.2 Transformer** — single-layer transformer (Q/K/V/O attention + MLP) built on the autodiff engine, trained on MNIST (28-row sequences) → 0.8537 test acc.
- [x] **PA1.3 Fused operators** — `MatMul+LayerNorm` and `MatMul+Softmax` fused forward/backward ops + write-up (`pa1/part3.txt`).
- [x] **PA2.1 Collective primitives** — `myAllreduce` (recursive) and `myAlltoall` implemented over MPI send/recv, benchmarked vs native MPI.
- [x] **PA2.2 Data-parallel split** — `split_train` across data-parallel groups.
- [x] **PA2.3 Layer init (`get_info`)** — model/data-parallel indexing, comm groups, FC in/out dims.
- [x] **PA2.4/2.5 Naive TP forward & backward comms** — `naive_collect_forward/backward_input/output` for the W_o layer.
- [x] **PA2 (bonus) tiled matmul** — CPU tiled GEMM reference (`matmul_tiled_cpu.py`); the graded Triton kernel is GPU-only and documented as a partial (see below).
- [x] **PA3.1 Mixture of Experts** — `ShardedLinear`, `MoE_TP` (tensor parallel) and `MoE_EP` (expert parallel, all-to-all) + benchmark & analysis.
- [x] **PA3.2 Scaling laws & cost analysis** — `model_training_cost_analysis_llama`, `get_optimal_N_D_from_cost`, and `model_training_cost_analysis_deepseek` (bonus) + `moe.md`, `my_model_config.json`.
- [x] **PA3.3 Speculative decoding** — target/draft init, `generate_draft_tokens`, vectorized verification, full `speculative_decode` loop; lossless vs baseline.
- [ ] **PA3.4 Argumentative essay** — a written (non-code) deliverable; out of scope for a code repo, intentionally omitted.

## Project structure

```
cse234-ml-data-systems/
├── pa1/                     # autodiff engine + transformer + fused ops
│   ├── auto_diff.py         # Node/Op graph, forward + reverse-mode backward
│   ├── fused_ops.py         # MatMul+LayerNorm, MatMul+Softmax fused ops
│   ├── transformer.py       # transformer classifier trained on MNIST
│   └── tests/               # 40 pytest cases (forward/backward/fused)
├── pa2/                     # MPI parallelism
│   ├── mpi_wrapper/comm.py  # myAllreduce / myAlltoall from scratch
│   ├── model/func_impl.py   # get_info + naive TP forward/backward comms
│   ├── data/…               # data-parallel split
│   └── matmul_tiled_cpu.py  # CPU tiled GEMM (Triton kernel is GPU-only)
├── pa3/
│   ├── part1/moe.py         # SimpleMoE, MoE_TP, MoE_EP + benchmark + analysis
│   ├── part2/…              # scaling-law / training-cost analysis
│   └── part3/speculative_decode.py
├── results/                 # measured evidence (logs, pytest output, benchmarks)
├── requirements.txt
└── LICENSE
```

## How to run

Python repos use the shared csdiy env (Python 3.11, CPU): `D:\Project\_csdiy\.venv-ml\Scripts\python.exe`.
An MPI runtime is required for PA2/PA3-part1 (MS-MPI on Windows, OpenMPI/MPICH on Linux/macOS).

```bash
python -m pip install -r requirements.txt   # or reuse the shared venv

# --- PA1: autodiff + fused ops (40 tests) and transformer training ---
cd pa1
OMP_NUM_THREADS=3 python -m pytest -v            # 40 passed
OMP_NUM_THREADS=3 python transformer.py          # trains on MNIST -> ~0.85 test acc

# --- PA2: parallelism (needs MPI; 8 cores) ---
cd ../pa2
python   -m pytest -l -v tests/test_data_split.py
mpiexec -n 8 python -m pytest -l -v --with-mpi tests/test_get_info.py
mpiexec -n 4 python -m pytest -l -v --with-mpi tests/test_transformer_forward.py
mpiexec -n 4 python -m pytest -l -v --with-mpi tests/test_transformer_backward.py
mpiexec -n 8 python mpi-test.py --test_case myallreduce   # custom vs MPI
mpiexec -n 8 python mpi-test.py --test_case myalltoall
python matmul_tiled_cpu.py                        # CPU tiled GEMM check

# --- PA3 part 1: MoE (TP/EP), 5 ranks ---
cd ../pa3/part1
mpiexec -n 5 python test_moe.py                   # Simple/EP/TP MoE all pass
mpiexec -n 4 python benchmark.py --sweep          # TP vs EP timing sweep

# --- PA3 part 2: scaling-law / cost analysis ---
cd ../part2
python model_training_cost_analysis.py --model_config llama_7b_config.json
python model_training_cost_analysis.py --model_config deepseek_v3_config.json
python model_training_cost_analysis.py --training_budget 5000000

# --- PA3 part 3: speculative decoding (downloads small Pythia models on first run) ---
cd ../part3
OMP_NUM_THREADS=3 python speculative_decode.py
```

## Verification

Every result above is reproduced by the commands above and captured under `results/`:

- `results/pa1/pytest_results.txt` — 40/40 autodiff + fused-op tests passed.
- `results/pa1/train_mnist.log` — 20-epoch transformer training, final test accuracy **0.8537**.
- `results/pa2/pytest_results.txt` — data-split (4), get_info (2), TP forward (2), TP backward (2), all passed.
- `results/pa2/myallreduce_8ranks.txt`, `myalltoall_8ranks.txt` — 100-run correctness + timing vs MPI.
- `results/pa2/matmul_tiled_cpu.txt` — tiled GEMM matches NumPy (max abs err ≤ 6.9e-5).
- `results/pa3/moe_tests.txt` — official `test_moe.py` (5 ranks): Simple / EP / TP MoE all passed, plus cross-checks.
- `results/pa3/moe_benchmark.txt` — TP-vs-EP forward timing sweep.
- `results/pa3/cost_analysis.txt` — Llama-7B (6.738B params, 0.898 TFLOPs, 0.386 GB), DeepSeek-V3, and A100 as optimal GPU under a $5M budget.
- `results/pa3/speculative_decode.log` — speculative output equals greedy baseline (lossless), with per-prompt speedup / acceptance.

### Documented partials

- **PA2 Part 1 (Triton matmul kernel).** The graded kernel is Triton-on-GPU and is benchmarked
  by the course specifically on a Colab **T4 GPU** (fp16). This machine is **CPU-only with no CUDA**,
  so the Triton kernel cannot be compiled or timed here. The Triton implementation lives in
  `pa2/matmul_triton.ipynb`; a **correctness-equivalent CPU tiled GEMM** (`pa2/matmul_tiled_cpu.py`)
  is provided and verified against NumPy so the tiling logic is demonstrably correct. Install
  `triton` on a CUDA box to run/benchmark the notebook kernel.
- **PA3 Part 3 speedup.** On CPU (no tensor cores, no batched-KV kernels) speculative decoding is
  correctness-preserving (`output == baseline`) but only sometimes faster (1.16× on the well-aligned
  prompt); average speedup < 1 on CPU because the extra draft-model forward passes are not amortized
  the way they are on GPU. This is expected on CPU and documented in the log — the *algorithm* is
  correct and lossless, which is the graded property.
- **PA3 Part 4 (essay).** A 500-word argumentative essay, not code — intentionally excluded from a code repo.

## Tech stack

Python 3.11 · PyTorch 2.12 (CPU) · NumPy · scikit-learn · mpi4py + MS-MPI · Hugging Face
Transformers (Pythia via EleutherAI) · pytest / pytest-mpi. Triton (GPU-only) for the PA2 kernel.

## Key ideas / what I learned

- **Reverse-mode autodiff from scratch** — building a `Node`/`Op` computational graph, topological
  ordering, and per-op analytic adjoints (including 3D matmul, softmax, layernorm broadcasting).
- **Operator fusion** — why matmul+normalize is memory-bandwidth bound, and how fusing removes a
  full DRAM round-trip of the intermediate on both forward and backward.
- **Collective communication** — implementing all-reduce and all-to-all over point-to-point MPI,
  and why the naive versions cost more than the vendor library.
- **Parallelism strategies** — data vs tensor-model vs expert parallelism, and the concrete comms
  each induces; measuring why EP crushes TP for sparse MoE routing.
- **Scaling laws & training cost** — turning a model config into parameter/FLOP/memory counts and
  solving Chinchilla-style for optimal `(N, D)` under a dollar budget.
- **Inference acceleration** — speculative decoding with a draft/target pair, vectorized
  verification, and the guarantee that accepted tokens exactly match greedy decoding.

## Credits & license

Based on the programming assignments of **CSE 234: Data Systems for Machine Learning** by
**Prof. Hao Zhang, UC San Diego (Winter 2025)** — official starter code:
[`hao-ai-lab/cse234-w25`](https://github.com/hao-ai-lab/cse234-w25). This repository is an
independent educational reimplementation; all course materials, datasets, and specifications belong
to their original authors. Original code in this repo is released under the [MIT License](LICENSE).
