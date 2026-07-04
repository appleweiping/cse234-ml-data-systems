"""
CSE234 PA3 Part 3 - Speculative Decoding (CPU-runnable standalone).

Same SpeculativeDecoder implemented in `PA3_Speculative_Decoding.ipynb`, extracted
into a script so it can run and be measured on the CPU-only grading machine. The
notebook defaults to pythia-1.4b/160m on CUDA; here we use small Pythia models
(same tokenizer family, so the compatibility assertion holds) that run on CPU:

    target = EleutherAI/pythia-410m   draft = EleutherAI/pythia-70m

Greedy speculative decoding: the draft proposes k tokens in one call, the target
verifies them all in a single forward pass, and we accept the longest matching
prefix (greedy verification) plus one bonus token from the target. Greedy
speculative decoding is guaranteed to produce the SAME text as greedy baseline
decoding -- we assert that as a correctness check.

Run:  python speculative_decode.py
"""
import os
import time
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class SpeculativeDecoder:
    def __init__(self, target_model_name: str, draft_model_name: str, device: str = "cpu"):
        self.device = device
        self.target_model, self.target_tokenizer = self.initialize_target_model(target_model_name)
        self.draft_model, self.draft_tokenizer = self.initialize_draft_model(draft_model_name)
        assert self.target_tokenizer.vocab == self.draft_tokenizer.vocab, "Tokenizers must be compatible"
        self._last_acceptance = 0.0

    def initialize_target_model(self, model_name: str):
        print(f"Loading target model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        model.to(self.device)
        model.eval()
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def initialize_draft_model(self, model_name: str):
        print(f"Loading draft model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        model.to(self.device)
        model.eval()
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    @torch.no_grad()
    def generate_draft_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                              num_speculative_tokens: int = 10) -> torch.Tensor:
        """Greedily generate num_speculative_tokens from the draft model in one call."""
        out = self.draft_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=num_speculative_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self.draft_tokenizer.pad_token_id,
        )
        return out[:, input_ids.shape[1]:]

    @torch.no_grad()
    def verify_tokens_vectorized(self, input_ids: torch.Tensor, draft_tokens: torch.Tensor,
                                 attention_mask: torch.Tensor) -> Tuple[List[int], int]:
        """Verify all draft tokens in a single target forward pass (greedy).

        Returns the accepted token ids (accepted draft prefix + one correction /
        bonus token from the target) and the index of the first rejected token.
        """
        k = draft_tokens.shape[1]
        full = torch.cat([input_ids, draft_tokens], dim=1)
        full_mask = torch.cat([attention_mask, torch.ones_like(draft_tokens)], dim=1)
        logits = self.target_model(full, attention_mask=full_mask).logits  # (1, L+k, V)

        L = input_ids.shape[1]
        # Position L-1 predicts draft[0], L predicts draft[1], ...
        pred = logits[:, L - 1: L - 1 + k, :].argmax(dim=-1)[0]  # (k,)
        draft = draft_tokens[0]

        accepted: List[int] = []
        accepted_position = k
        for i in range(k):
            if int(pred[i]) == int(draft[i]):
                accepted.append(int(draft[i]))
            else:
                accepted.append(int(pred[i]))   # target correction token
                accepted_position = i
                break
        else:
            # All k accepted -> one extra token from the target.
            bonus = int(logits[:, L + k - 1, :].argmax(dim=-1)[0])
            accepted.append(bonus)
        return accepted, accepted_position

    @torch.no_grad()
    def speculative_decode(self, prompt: str, max_tokens: int = 100,
                           num_speculative_tokens: int = 8) -> str:
        inputs = self.target_tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        prompt_length = input_ids.shape[1]

        total_tokens_generated = prompt_length
        total_draft_tokens_proposed = 0
        total_draft_tokens_accepted = 0
        eos = self.target_tokenizer.eos_token_id

        while total_tokens_generated - prompt_length < max_tokens:
            draft_tokens = self.generate_draft_tokens(
                input_ids, attention_mask, num_speculative_tokens
            )
            k = draft_tokens.shape[1]
            total_draft_tokens_proposed += k

            accepted, first_reject = self.verify_tokens_vectorized(
                input_ids, draft_tokens, attention_mask
            )
            total_draft_tokens_accepted += min(first_reject, k)

            new = torch.tensor([accepted], device=self.device, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, new], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(new)], dim=1)
            total_tokens_generated = input_ids.shape[1]

            if eos is not None and eos in accepted:
                break

        acceptance_rate = (total_draft_tokens_accepted / total_draft_tokens_proposed
                           if total_draft_tokens_proposed > 0 else 0)
        self._last_acceptance = acceptance_rate
        # A speculative round appends a variable number of tokens (accepted prefix
        # + bonus), so the loop can overshoot the budget. Truncate to exactly
        # max_tokens new tokens so the output is comparable to greedy baseline
        # decoding (which stops at exactly max_new_tokens).
        input_ids = input_ids[:, : prompt_length + max_tokens]
        return self.target_tokenizer.decode(input_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def baseline_decode(self, prompt: str, max_tokens: int = 100) -> Tuple[str, float]:
        inputs = self.target_tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        start = time.time()
        out = self.target_model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_tokens, do_sample=False, num_beams=1,
            pad_token_id=self.target_tokenizer.pad_token_id,
        )
        elapsed = time.time() - start
        return self.target_tokenizer.decode(out[0], skip_special_tokens=True), elapsed


def main():
    torch.manual_seed(0)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    decoder = SpeculativeDecoder(
        target_model_name="EleutherAI/pythia-410m",
        draft_model_name="EleutherAI/pythia-70m",
        device=device,
    )

    prompts = [
        "The future of Artificial Intelligence is",
        "Once upon a time, in a small village,",
    ]
    max_tokens = 60
    k = 8

    print("\n" + "=" * 72)
    print(f"Speculative decoding  (target=pythia-410m, draft=pythia-70m, "
          f"device={device}, k={k}, max_tokens={max_tokens})")
    print("=" * 72)

    spec_times, base_times, acc_rates, matches = [], [], [], []
    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i + 1}: {repr(prompt)} ---")
        t0 = time.time()
        spec_text = decoder.speculative_decode(prompt, max_tokens=max_tokens,
                                               num_speculative_tokens=k)
        spec_elapsed = time.time() - t0
        acc_rates.append(decoder._last_acceptance)
        spec_times.append(spec_elapsed)

        base_text, base_elapsed = decoder.baseline_decode(prompt, max_tokens=max_tokens)
        base_times.append(base_elapsed)

        match = spec_text.strip() == base_text.strip()
        matches.append(match)
        print(f"baseline {base_elapsed:.2f}s | speculative {spec_elapsed:.2f}s | "
              f"speedup {base_elapsed / spec_elapsed:.2f}x | "
              f"acceptance {decoder._last_acceptance:.2%} | output==baseline: {match}")

    print("\n" + "=" * 72)
    avg_spec = sum(spec_times) / len(spec_times)
    avg_base = sum(base_times) / len(base_times)
    avg_acc = sum(acc_rates) / len(acc_rates)
    print(f"AVERAGE  baseline {avg_base:.2f}s  speculative {avg_spec:.2f}s  "
          f"speedup {avg_base / avg_spec:.2f}x  acceptance {avg_acc:.2%}  "
          f"all_correct={all(matches)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
