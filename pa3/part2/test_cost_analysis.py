"""Sanity checks for the PA3 Part 2 cost-analysis functions.

These assert against publicly known figures (Llama-7B ~6.7B params, DeepSeek-V3
671B params) and internal consistency of the scaling-law optimizer.

Run:  python -m pytest -q test_cost_analysis.py
"""
import os

from model_training_cost_analysis import (
    model_training_cost_analysis_llama,
    model_training_cost_analysis_deepseek,
    get_optimal_N_D_from_cost,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def test_llama_params():
    n, tflops, mem = model_training_cost_analysis_llama(
        os.path.join(HERE, "llama_7b_config.json")
    )
    # Llama-7B has ~6.7B parameters.
    assert 6.6e9 < n < 6.8e9, n
    assert tflops > 0
    assert mem > 0


def test_deepseek_params():
    n, tflops, mem = model_training_cost_analysis_deepseek(
        os.path.join(HERE, "deepseek_v3_config.json")
    )
    # DeepSeek-V3 has ~671B total parameters.
    assert 6.6e11 < n < 6.8e11, n
    assert tflops > 0
    assert mem > 0


def test_optimal_allocation():
    N, D, flops, gpu = get_optimal_N_D_from_cost(5_000_000)
    # A100 is the most FLOPs-per-dollar of the three options.
    assert gpu == "A100"
    assert flops > 0
    # Chinchilla-style: C ~= 6*N*D should recover the budget FLOPs.
    assert abs(6 * N * D - flops) / flops < 0.05
    # Sensible ranges for a ~$5M budget.
    assert 1e10 < N < 1e11, N
    assert 1e12 < D < 1e13, D


if __name__ == "__main__":
    test_llama_params()
    test_deepseek_params()
    test_optimal_allocation()
    print("All cost-analysis checks passed.")
