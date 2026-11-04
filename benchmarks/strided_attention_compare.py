"""Compare MHA, GQA, and VHA on packed shifted-window attention.

The default shape is the packed work for one 4k-token sample with a 128-token
window (264 windows is 22 shifted depths).  This measures the attention block
including QKV/output projections and backward, rather than only SDPA.

Examples:
    CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/strided_attention_compare.py
    CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/strided_attention_compare.py \
        --mode gqa --kv-heads 2
"""

from __future__ import annotations

import argparse
import math
import time

import torch
from torch.nn import functional as F


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("mha", "gqa", "vha", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--windows", type=int, default=264)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=1280)
    parser.add_argument("--heads", type=int, default=20)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--postmix", choices=("runtime", "none"), default="runtime",
        help="VHA low-rank Postmix; use none to isolate Premix cost.",
    )
    return parser


def _rms_norm(x):
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6).to(x.dtype)


def _run(mode, args):
    device = torch.device(args.device)
    dtype = torch.bfloat16
    hidden = args.hidden_size
    heads = args.heads
    kv_heads = heads if mode == "mha" else args.kv_heads
    if hidden % heads:
        raise ValueError("hidden-size must be divisible by heads")
    head_dim = hidden // heads
    if mode == "vha" and heads % (2 * kv_heads):
        raise ValueError("VHA requires heads divisible by 2 * kv-heads")

    physical_q_heads = heads // 2 if mode == "vha" else heads
    q_width = physical_q_heads * head_dim
    kv_width = kv_heads * head_dim
    x = torch.randn(
        args.windows, args.window, hidden, device=device, dtype=dtype,
    )
    weights = [
        torch.randn(width, hidden, device=device, dtype=dtype, requires_grad=True)
        for width in (q_width, kv_width, kv_width)
    ]
    output_weight = torch.randn(
        hidden, hidden, device=device, dtype=dtype, requires_grad=True,
    )
    parameters = [*weights, output_weight]

    premix = None
    postmix_a = postmix_b = None
    if mode == "vha":
        # SelfAttentionVHA (PaddlePaddle/VHA, commit ab7ceded) initializes
        # the full-rank Premix close to identity and uses a small Normal
        # Postmix U; V is zero so the initial forward path is unchanged.
        premix = torch.eye(head_dim, device=device, dtype=dtype).expand(
            kv_heads, physical_q_heads, head_dim, head_dim,
        ).clone()
        premix.add_(torch.randn_like(premix) * (0.1 / math.sqrt(head_dim)))
        premix.requires_grad_()
        parameters.append(premix)
        if args.postmix == "runtime":
            rank = max(1, heads // 4)
            postmix_a = torch.randn(
                heads, rank, device=device, dtype=dtype,
            ) * 0.01
            postmix_a.requires_grad_()
            postmix_b = torch.zeros_like(postmix_a, requires_grad=True)
            parameters.extend((postmix_a, postmix_b))
    def step():
        q = F.linear(x, weights[0]).view(
            args.windows, args.window, physical_q_heads, head_dim,
        )
        k = F.linear(x, weights[1]).view(
            args.windows, args.window, kv_heads, head_dim,
        )
        v = F.linear(x, weights[2]).view(
            args.windows, args.window, kv_heads, head_dim,
        )
        if mode == "vha":
            # Each physical query head is premixed independently for every
            # KV group, before QK normalization and RoPE as in the paper. The
            # virtual heads are then flattened for SDPA.
            q = torch.einsum("nwgd,kgde->nwkge", q, premix).reshape(
                args.windows, args.window, heads, head_dim,
            )
        q = _rms_norm(q)
        k = _rms_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=(kv_heads != heads),
        ).transpose(1, 2)
        if mode == "vha" and args.postmix == "runtime":
            mixed = torch.einsum("nwhd,hr->nwrd", y, postmix_a)
            y = y + torch.einsum("nwrd,hr->nwhd", mixed, postmix_b)
        return F.linear(y.reshape(-1, hidden), output_weight).sum()

    for _ in range(args.warmup):
        for parameter in parameters:
            parameter.grad = None
        step().backward()
    torch.cuda.synchronize(device)
    timings = []
    for _ in range(args.steps):
        for parameter in parameters:
            parameter.grad = None
        started = time.perf_counter()
        step().backward()
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)
    median = sorted(timings)[len(timings) // 2]
    relative = (
        "n/a" if _run.mha_median is None
        else f"{median / _run.mha_median:.3f}"
    )
    print(
        f"{mode:>3}  qkv_width={q_width + 2 * kv_width:4d} "
        f"windows={args.windows} window={args.window} "
        f"median_fwd_bwd_ms={median * 1e3:.3f} "
        f"mean_fwd_bwd_ms={sum(timings) / len(timings) * 1e3:.3f} "
        f"relative_to_mha={relative}"
    )
    return median


def main():
    args = _parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; select one GPU, e.g. CUDA_VISIBLE_DEVICES=1")
    torch.set_float32_matmul_precision("high")
    modes = ("mha", "gqa", "vha") if args.mode == "all" else (args.mode,)
    _run.mha_median = None
    for mode in modes:
        median = _run(mode, args)
        if mode == "mha":
            _run.mha_median = median


if __name__ == "__main__":
    main()
