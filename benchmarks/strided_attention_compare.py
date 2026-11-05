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
from torch.nn.attention import SDPBackend, sdpa_kernel


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
        "--compile", action="store_true",
        help="Compile the fixed-shape attention block with torch.compile.",
    )
    parser.add_argument(
        "--compile-mode", choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"),
        default="default", help="Inductor mode for the compiled path.",
    )
    parser.add_argument(
        "--fullgraph", action="store_true",
        help="Require one complete Dynamo graph for the compiled path.",
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Print the CUDA profiler table for one measured step.",
    )
    parser.add_argument(
        "--fused-qkv", action="store_true",
        help="Use one fused QKV projection instead of three linears.",
    )
    parser.add_argument(
        "--fused-qk-norm", action="store_true",
        help="Normalize Q and K in one concatenated RMSNorm operation.",
    )
    parser.add_argument(
        "--native-rms", action="store_true",
        help="Use PyTorch's fused RMSNorm primitive in the fixed-shape test.",
    )
    parser.add_argument(
        "--bf16-rms", action="store_true",
        help="Accumulate RMSNorm statistics in BF16 (diagnostic only).",
    )
    parser.add_argument(
        "--postmix", choices=("runtime", "none"), default="runtime",
        help="VHA low-rank Postmix; use none to isolate Premix cost.",
    )
    parser.add_argument(
        "--explicit-gqa", action="store_true",
        help="Diagnostic: repeat K/V heads before SDPA instead of native GQA.",
    )
    parser.add_argument(
        "--low-level-flash", action="store_true",
        help="Use ATen's FlashAttention operator directly on CUDA.",
    )
    parser.add_argument(
        "--varlen-flash", action="store_true",
        help="Use the packed varlen FlashAttention operator for CUDA.",
    )
    parser.add_argument(
        "--sdpa-backend",
        choices=("default", "flash", "efficient", "cudnn", "math"),
        default="default",
        help="Select a fixed SDPA backend; default lets PyTorch choose.",
    )
    return parser


def _rms_norm(x):
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6).to(x.dtype)


def _rms_norm_bf16(x):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)


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
    if args.fused_qkv:
        qkv_weight = torch.randn(
            q_width + 2 * kv_width, hidden, device=device, dtype=dtype,
            requires_grad=True,
        )
        weights = [qkv_weight]
    else:
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
            kv_heads, head_dim, head_dim,
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
    def compute():
        if args.fused_qkv:
            qkv = F.linear(x, weights[0])
            q, k, v = qkv.split((q_width, kv_width, kv_width), dim=-1)
        else:
            q = F.linear(x, weights[0])
            k = F.linear(x, weights[1])
            v = F.linear(x, weights[2])
        q = q.view(args.windows, args.window, physical_q_heads, head_dim)
        k = k.view(args.windows, args.window, kv_heads, head_dim)
        v = v.view(args.windows, args.window, kv_heads, head_dim)
        if mode == "vha":
            # Each physical query head is premixed independently for every
            # KV group, before QK normalization and RoPE as in the paper. The
            # virtual heads are then flattened for SDPA.
            q = torch.einsum("nwgd,kde->nwkge", q, premix).reshape(
                args.windows, args.window, heads, head_dim,
            )
        norm = (
            lambda z: F.rms_norm(z, (z.size(-1),), eps=1e-6)
            if args.native_rms else _rms_norm
        )
        if args.bf16_rms:
            norm = _rms_norm_bf16
        if args.fused_qk_norm:
            qk = norm(torch.cat((q, k), dim=2))
            q, k = qk.split((physical_q_heads, kv_heads), dim=2)
        else:
            q = norm(q)
            k = norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        use_native_gqa = kv_heads != heads and not args.explicit_gqa
        if kv_heads != heads and args.explicit_gqa:
            repeat = heads // kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        use_varlen_flash = args.varlen_flash and q.device.type == "cuda"
        use_low_level_flash = args.low_level_flash and q.device.type == "cuda"
        if use_varlen_flash:
            total = args.windows * args.window
            cu = torch.arange(
                0, total + 1, args.window, device=q.device, dtype=torch.int32
            )
            y, *_ = torch.ops.aten._flash_attention_forward(
                q.transpose(1, 2).reshape(total, heads, head_dim),
                k.transpose(1, 2).reshape(total, kv_heads, head_dim),
                v.transpose(1, 2).reshape(total, kv_heads, head_dim),
                cu, cu, args.window, args.window, 0.0, True, False,
            )
            y = y.view(args.windows, args.window, heads, head_dim)
        elif use_low_level_flash:
            y, *_ = torch.ops.aten._scaled_dot_product_flash_attention.default(
                q, k, v, 0.0, True, False,
            )
        elif args.sdpa_backend == "default":
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=use_native_gqa,
            )
        else:
            backend = {
                "flash": SDPBackend.FLASH_ATTENTION,
                "efficient": SDPBackend.EFFICIENT_ATTENTION,
                "cudnn": SDPBackend.CUDNN_ATTENTION,
                "math": SDPBackend.MATH,
            }[args.sdpa_backend]
            with sdpa_kernel(backend):
                y = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, enable_gqa=use_native_gqa,
                )
        y = y.transpose(1, 2)
        if mode == "vha" and args.postmix == "runtime":
            mixed = torch.einsum("nwhd,hr->nwrd", y, postmix_a)
            y = y + torch.einsum("nwrd,hr->nwhd", mixed, postmix_b)
        return F.linear(y.reshape(-1, hidden), output_weight).sum()

    # The packed benchmark has a fixed [windows, window] shape. Compiling the
    # complete projection/norm/SDPA/output block lets Inductor fuse the small
    # elementwise operations around SDPA. The dynamic full LoL dispatch path
    # must keep its eager block boundary because its routed window counts vary.
    step = (
        torch.compile(
            compute, dynamic=False, mode=args.compile_mode,
            fullgraph=args.fullgraph,
        )
        if args.compile else compute
    )

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
        if args.profile and not timings:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,
            ) as prof:
                step().backward()
            print(prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=20
            ))
        else:
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
    if args.sdpa_backend != "default":
        selected = {
            "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "cudnn": SDPBackend.CUDNN_ATTENTION,
            "math": SDPBackend.MATH,
        }[args.sdpa_backend]
        # Fixed-shape compilation captures this setting cleanly. The
        # dynamically routed model keeps the default selection because
        # backend guards and changing packed shapes add graph breaks there.
        torch.backends.cuda.enable_flash_sdp(
            selected is SDPBackend.FLASH_ATTENTION
        )
        torch.backends.cuda.enable_mem_efficient_sdp(
            selected is SDPBackend.EFFICIENT_ATTENTION
        )
        torch.backends.cuda.enable_cudnn_sdp(
            selected is SDPBackend.CUDNN_ATTENTION
        )
        torch.backends.cuda.enable_math_sdp(selected is SDPBackend.MATH)
    torch.set_float32_matmul_precision("high")
    modes = ("mha", "gqa", "vha") if args.mode == "all" else (args.mode,)
    _run.mha_median = None
    for mode in modes:
        median = _run(mode, args)
        if mode == "mha":
            _run.mha_median = median


if __name__ == "__main__":
    main()
