"""Forward+backward benchmark for the 600M shifted-window model.

The ``padded`` mode is the original one-expert-at-a-time dispatch.  The
``grouped`` mode is the production path used by the LoL dynamic-window config.
Select one GPU explicitly on this host, for example ``CUDA_VISIBLE_DEVICES=1``
for the GB300.
"""

from __future__ import annotations

import argparse
import gc
import pathlib
import sys
import time

import torch

# Make ``python benchmarks/lol_step_baseline.py`` work from the repository
# root without requiring a PYTHONPATH override.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from model.factory import build_model


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("grouped", "padded", "all"), default="grouped")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--kv-heads", type=int, default=None,
        help="Train a GQA variant with this many KV heads; null keeps MHA.",
    )
    parser.add_argument("--experts", type=int, default=24)
    return parser


def _run(mode, args):
    config = dict(
        arch="lol_dynamic_window",
        routed=True,
        attention_strategy="shifted_window",
        n_head=20,
        n_kv_head=args.kv_heads,
        n_embd=1280,
        n_experts=args.experts,
        window=128,
        window_stride=64,
        first_window_routing="learned_token_depth",
        first_window_length=None,
        recurrent_depth=24,
        n_prelude_layers=1,
        n_coda_layers=1,
        depth_expert_film=False,
        weight_tie_router=True,
        aux_loss_coeff=0.02,
        grouped_window_dispatch=mode == "grouped",
        block_size=args.sequence_length,
        vocab_size=50432,
        normalization="affine_rmsnorm",
        qk_norm=True,
        recurrent_input_injection="none",
        moe_ffn=False,
        moe_ffn_scope="all",
        moe_ffn_n_experts=8,
        moe_ffn_top_k=2,
        moe_ffn_hidden_dim=None,
        moe_ffn_aux_loss_coeff=0.01,
        sandwich_norm=True,
        recurrent_state_std=1.0,
    )
    model, _ = build_model(config)
    model.to(args.device).train()
    if not args.no_compile:
        model.compile_blocks()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    idx = torch.randint(
        1, config["vocab_size"],
        (args.batch, args.sequence_length),
        device=args.device,
    )
    idx[:, -1] = 0
    targets = torch.roll(idx, -1, dims=1)

    def step():
        model.zero_grad(set_to_none=True)
        with torch.autocast(args.device, dtype=torch.bfloat16):
            loss = model(
                idx, targets=targets, return_router_diagnostics=False,
            )[0]
        loss.backward()
        torch.cuda.synchronize(args.device)

    for _ in range(args.warmup):
        step()
    timings = []
    for _ in range(args.steps):
        started = time.perf_counter()
        step()
        timings.append(time.perf_counter() - started)
    median = sorted(timings)[len(timings) // 2]
    tokens_per_second = args.batch * args.sequence_length / median
    print(
        f"{mode:>7} params={parameter_count / 1e6:.1f}M "
        f"batch={args.batch} seq={args.sequence_length} "
        f"median_fwd_bwd_ms={median * 1e3:.3f} "
        f"tokens_per_second={tokens_per_second:.0f}"
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return median


def main():
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; select one GPU, e.g. CUDA_VISIBLE_DEVICES=1")
    torch.set_float32_matmul_precision("high")
    modes = ("grouped", "padded") if args.mode == "all" else (args.mode,)
    results = {mode: _run(mode, args) for mode in modes}
    if len(results) == 2:
        print(f"padded/grouped speedup={results['padded'] / results['grouped']:.2f}x")


if __name__ == "__main__":
    main()
