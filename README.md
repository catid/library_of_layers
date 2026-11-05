# Library of Layers (LoL)

Code to reproduce **Library of Layers (LoL): Adaptively Ordering Transformer Layers at Runtime**. A joint project by Abdellah Rahmani\*, Sean McLeish\*, Ang Li, Sanae Lotfi, Pascal Frossard, and Micah Goldblum.

\* Equal contribution.

<p align="center">
  <a href="https://arxiv.org/abs/2611.03526">
    <img src="docs/media/paper.svg" alt="Paper on arXiv" width="112" height="38">
  </a>
</p>

<p align="center">
  <img src="docs/media/LoL_teaser.gif" alt="Animated overview of Library of Layers (LoL)" width="960">
</p>

# Citing Our Work
```
@article{rahmani2026library,
    title={Library of Layers (LoL): Adaptively Ordering Transformer Layers at Runtime}, 
    author={Abdellah Rahmani and Sean McLeish and Ang Li and Sanae Lotfi and Pascal Frossard and Micah Goldblum},
    journal={arXiv preprint arXiv:2611.03526},
    year={2026}
}
```

# Reproduction Instructions
Use Python 3.12 and the dependencies in [pyproject.toml](pyproject.toml).

## Data preparation

Training requires prepared corpora; select their location with `data.data_dir`.
Use the preparation modules' `--help` for source, quota, and output options:

| Corpus | Preparation entry point |
| --- | --- |
| FineWeb-Edu | `python -m data.prepare_fineweb_edu_30b --help` |
| Math | `python -m data.prepare_math --help` |
| Multilingual FineWeb2 | `python -m data.prepare_fineweb2_multilingual --help` |
| Safety | `python -m data.prepare_fineweb_safety --help` |
| Sudoku | Generate with `python -m data.build --sudoku --help`, then pack with `python -m data.prepare_sudoku --help`. |
| Maze | Generate with `python -m data.build --maze --help`, then pack with `python -m data.prepare_maze --help`. |

For Sudoku and maze, point training/evaluation at the packed output and use the
corresponding shell recipe's matching label masks and first-window settings.

## Training and Evaluation
The commands below reproduce the paper's figures and results. Each training workflow owns its plotting commands and has
`TRAIN`, `EVALUATE`, and `PLOT` sections. Run the commands you need from the
repository root; comments identify shared runs used by several figures.

| Shell file | What it reproduces |
| --- | --- |
| [math.sh](shells/math.sh) | All math results: five-model training/evaluation, main and extended loss/FLOP figures, expert utilization, and benchmark tables. |
| [lol_vs_huginn.sh](shells/lol_vs_huginn.sh) | LoL versus Huginn: validation loss, training loss over FLOPs, and expert utilization across seeds. |
| [lol_vs_moe.sh](shells/lol_vs_moe.sh) | LoL versus mixture-of-experts models, including expert utilization across seeds. |
| [safety_probes.sh](shells/safety_probes.sh) | Safety prediction from router probabilities. |
| [lol_backprop_depth.sh](shells/lol_backprop_depth.sh) | Growing depth and backpropagation ablation. |
| [lol_load_balancing.sh](shells/lol_load_balancing.sh) | Loss-free versus auxiliary-loss load balancing. |
| [lol_router_sharing.sh](shells/lol_router_sharing.sh) | Tied versus untied router weights. |
| [vanilla_attention.sh](shells/vanilla_attention.sh) | Full, sliding-window, and shifted-window attention in dense models. |
| [multilingual.sh](shells/multilingual.sh) | Multilingual routing progression and the combined language/Sudoku/maze routing overview. |
| [sudoku.sh](shells/sudoku.sh) | Sudoku strategy adherence, solution accuracy, routing agreement, and routing paths. |
| [maze.sh](shells/maze.sh) | Maze strategy and solution generation, routing agreement, and routing paths. |

Before evaluation, set the `CHECKPOINT_*` variables shown in the script to the corresponding trained checkpoints, for example:

```bash
export CHECKPOINT_sudoku_lol=/path/to/checkpoint.pt
```

## Strided-attention benchmarks

### Context-length scaling: LoL, SWA, and full FlashAttention

For a fixed local width `w=128`, LoL shifted-window attention and standard
sliding-window attention (SWA) both scale linearly with context length `S`:
each token participates in a bounded local interaction. Full FlashAttention
has no sparsity and scales quadratically, so doubling context roughly doubles
the work for LoL/SWA but quadruples the attention interaction count for full
attention. The crossover becomes more pronounced at 8k and 16k context even
when full FlashAttention remains the fastest dense kernel at short context.

LoL and SWA do not have identical work. LoL partitions each depth into
non-overlapping fixed windows and shifts the partition at later depths. SWA
uses overlapping trailing windows at every layer. For the same causal width,
the nominal per-token interaction count is similar, but LoL avoids repeatedly
materializing overlapping windows and gets a regular packed shape that maps
well to FlashAttention. Shifts provide cross-window communication over depth;
SWA provides it immediately through the trailing neighborhood. LoL therefore
usually has the lower kernel constant, while SWA has the simpler connectivity
semantics and is often easier to implement with a standard local-attention
kernel.

The packed kernel benchmark can measure LoL's attention work at several
contexts by scaling the packed window count from the validated 4k shape:

```bash
for s in 4096 8192 16384; do
  CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/strided_attention_compare.py \
    --mode gqa --kv-heads 2 --fused-qkv --native-rms --compile \
    --sdpa-backend flash --windows $((264 * s / 4096)) \
    --warmup 8 --steps 20
done
```

Those measurements are attention-block timings, not full-model training
times. The validated 4k point is **1.264 ms** for GQA-2, versus **3.58 ms**
for the original packed MHA baseline. The full B8/S4096 LoL step is **255.1
ms** because routed GEMMs and dispatch dominate outside attention. A fair SWA
comparison should use the same hidden size, head layout, local width, number
of layers, causal mask, and forward/backward measurement; comparing LoL's
packed windows with a dense full-context SWA implementation would conflate
the attention pattern with kernel and packing overhead.

### GDN2 → LoL → LoL → FA layer schedule

We also measured the proposed four-layer schedule at B1/S4096 on the GB300,
using D=1280, BF16, forward plus backward, and eight timed steps after warmup.
The GDN2 reference is the available cuDNN 1.32 Frost implementation; its
Triton backend is included as a cross-check. The LoL and FA figures are the
attention-core benchmark at the same hidden size and sequence length.

| Layer/core | Backend and shape | Median step |
| --- | --- | ---: |
| GDN2 | cuDNN 1.32 Frost, 20 heads, chunk size 64 | **5.31 ms** |
| GDN2 | Triton, 20 heads, chunk size 64 | **5.61 ms** |
| LoL local attention | GQA-2, 264 packed 128-token windows | **0.836 ms** |
| Dense FA | latest cuDNN SDPA, 4096-token causal sequence | **0.839 ms** |
| Dense FA | PyTorch FlashAttention backend, same shape | **1.07 ms** |

These are not full transformer-layer timings: the LoL and FA entries isolate
the attention core, whereas GDN2 includes its recurrent state update,
projections, gates, normalization, and output projection. On this basis, one
`GDN2 → LoL → LoL → FA` group is approximately **7.82 ms** of measured core
work (5.31 + 0.836 + 0.836 + 0.839), or about **1.96 ms per layer** when
amortized over four layers. The schedule therefore spends most of its layer
budget on GDN2, while LoL and cuDNN FA are each roughly six times cheaper per
core invocation. GDN2 is the long-range state carrier; LoL supplies cheap
shifted local processing; FA provides periodic exact global interaction.

To reproduce the LoL and FA measurements:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/strided_attention_compare.py \
  --mode gqa --kv-heads 2 --fused-qkv --native-rms --compile \
  --sdpa-backend flash --windows 264 --window 128 --warmup 8 --steps 20

CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/strided_attention_compare.py \
  --mode mha --fused-qkv --native-rms --compile \
  --sdpa-backend cudnn --windows 1 --window 4096 --warmup 8 --steps 20
```

The GDN2 measurement uses the machine-local GDN2/cuDNN benchmark because that
implementation is not yet part of this LoL repository; its environment must
include the cuDNN 1.32 package and the CUTLASS DSL extension.

The composed repeat was also benchmarked directly, including FFN and residual
work in every layer. The benchmark uses `FF=12000`, producing **600.08M
parameters**:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/gdn2_lol_fa_repeat.py
```

At B1/S4096, BF16, AdamW forward plus backward, and five timed steps after two
warmups, the actual repeat measured **17.33 ms** (236.4k tokens/s) with the
reference GDN2 path. This is
higher than the 13.69 ms sum of isolated core timings because it includes four
FFNs, residuals, normalizations, projections, and optimizer work. The composed
implementation is `GDN2(cuDNN) → LoL(GQA-4, local FlashAttention) →
LoL(GQA-4, local FlashAttention) → FA(cuDNN)`.

Per-layer CUDA-event hooks measure where that training time goes:

| Layer | Forward | Backward | Total | Share of repeat |
| --- | ---: | ---: | ---: | ---: |
| GDN2 | 2.96 ms | 3.59 ms | **6.55 ms** | **37.1%** |
| LoL-1 | 0.84 ms | 2.05 ms | **2.88 ms** | **16.3%** |
| LoL-2 | 0.84 ms | 2.02 ms | **2.86 ms** | **16.2%** |
| FA | 0.96 ms | 2.21 ms | **3.17 ms** | **17.9%** |
| Output/optimizer remainder | — | — | **~2.20 ms** | **~12.5%** |

The percentages are based on the measured **17.66 ms** median step from the
hook-instrumented run; hooks add a small amount of scheduling overhead, so the
individual totals do not need to sum exactly to the uninstrumented 17.33 ms
measurement. GDN2 is the largest single cost, while the two LoL layers together
consume about **32.5%** and FA about **17.9%**. Reproduce this breakdown with:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/gdn2_lol_fa_repeat_timed.py
```

The benchmark now defaults to the optimized `catid/gdn2` checkout at commit
`11d9f76`, with fused GDN2 projections and recompute disabled. On the same
machine this configuration measured **17.38 ms** and produced the following
shares: GDN2 **36.3%**, LoL-1 **16.6%**, LoL-2 **16.5%**, FA **18.3%**, and
the remainder **~12.3%**. That is about **1.7% faster** than the reference
GDN2 configuration; the improvement comes from GDN2 forward/projection work,
while its backward remains the largest part of that layer.

For a width-matched 600M-class comparison (`D=3072`, 48 heads, head dimension
64), the same B1/S4096 measurements were: cuDNN GDN2 **5.74 ms**, LoL GQA-4
local attention **3.42 ms**, and latest cuDNN dense FA **1.12 ms**. A
width-matched `GDN2 → LoL → LoL → FA` repeat is therefore approximately
**13.69 ms** of core work, or **3.42 ms per layer averaged over the group**.
This is the more conservative 600M-width estimate; the D=1280 table above is
useful for the current LoL implementation's actual hidden size.

The 600M-class grouped shifted-window path can be measured on one GPU with:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/lol_step_baseline.py \
  --mode all --batch 1 --sequence-length 4096
```

The benchmark measures forward plus backward for the 592.9M configuration. A
GB300 B1 run measured 158.8 ms per step for grouped dispatch. Batch 1 per GPU
is therefore a practical starting point for an 8-GPU run with gradient
accumulation.

For a matched 8-GPU-style microbatch, use `--batch 8`. A clean GB300 run
(`--warmup 5 --steps 10`, sequence length 4096) measured **272.7 ms** for
grouped MHA dispatch and **1081.4 ms** for the original padded dispatch
(**3.97x**). The grouped path processes about 120k tokens/s at this batch size.
For the uncompiled baseline (`--no-compile --warmup 3 --steps 5`), the same
run measured 705.0 ms grouped and 2407.2 ms padded. Per-block compilation is
therefore a separate roughly 2.6x improvement for grouped dispatch.
To keep a GQA comparison near the same parameter count, use 29 experts with
two KV heads:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/lol_step_baseline.py \
  --mode grouped --batch 8 --sequence-length 4096 \
  --kv-heads 2 --experts 29
```

That configuration is 603.1M parameters; its native `enable_gqa` baseline was
**262.6 ms** per step on the same GB300. The direct FlashAttention path below
is faster. A GQA
run with the original 24 experts is only 516.2M parameters, so its larger
speedup is not a fair 600M comparison.
With one KV head and 29 experts, the model is 598.0M parameters and measured
**255.7 ms** with direct FlashAttention; routed FFN/GEMM work dominates
the end-to-end result.

The grouped GQA path now uses the low-level Blackwell FlashAttention operator
for compact K/V tensors, while retaining the public SDPA fallback on CPU and
for MHA. In a matched B8/S4096 run this reduced fair GQA-2 from **265.1 ms**
with native `enable_gqa` to **254.6 ms** (about **1.04x**). The gain is small
end to end because routed GEMMs dominate, but it removes the GQA wrapper from
the actual strided attention call.

To compare attention head layouts on the packed 128-token windows, run:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/strided_attention_compare.py
```

The packed attention benchmark also has a fixed-shape compiler path:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/strided_attention_compare.py \
  --compile --warmup 8 --steps 30
```

On the GB300 this reduces MHA packed attention from 3.58 ms to 1.65 ms per
forward/backward step (2.17x). It is a kernel benchmark optimization; applying
the compiled function inside the dynamically routed full model adds graph-call
overhead and does not improve the end-to-end step.

The fixed-shape benchmark also accepts `--compile-mode max-autotune`. With
GQA-2 and explicit FlashAttention, this measured **1.27 ms** per
forward/backward step (versus 1.37 ms with the default compile mode). The
autotuning cost is paid at compilation time, so this setting is intended for
long-running training jobs with a stable window shape.

Passing `--fused-qkv` combines the Q, K, and V projections into one GEMM before
the attention kernel. Together with `--compile-mode max-autotune`, GQA-2
measured **1.21 ms** on the same workload. The grouped production path already
has a grouped QKV projection, so this benchmark option measures the equivalent
fixed-shape projection schedule.

Packed varlen FlashAttention was tested with the same 264 windows and measured
**1.30 ms**, slower than the regular fixed-window kernel because every window
already has the same length. BF16 RMS accumulation also showed no measurable
gain.
An explicit bounded-window varlen formulation was slower again at **1.32 ms**.

A matched single-process run using fused QKV, native RMSNorm, and explicit
FlashAttention measured **1.395 ms** for MHA and **1.261 ms** for GQA-2
(1.11x). VHA measured **2.559 ms** with runtime Postmix, confirming that its
extra mixing work outweighs the narrower projection in this training kernel.
Calling the ATen FlashAttention operator directly for MHA measured **1.401 ms**
versus **1.405 ms** through the public SDPA wrapper, so the direct call is
effectively neutral there as well.
The cuDNN SDPA backend measured **1.408 ms** for GQA-2, slower than the
1.261 ms FlashAttention result.

For this fixed shape, selecting the backend explicitly gives another isolated
kernel improvement:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/strided_attention_compare.py \
  --mode mha --compile --sdpa-backend efficient --warmup 8 --steps 30
```

The GB300 measurement was **1.35 ms** for MHA, versus **1.64 ms** with the
default backend. Explicit Flash was **1.41 ms**. Native GQA-2 with explicit
Flash was **1.35 ms**. These settings are exposed for the fixed-shape kernel
benchmark; forcing them inside dynamic routed dispatch caused graph breaks and
was slower, so the production model keeps PyTorch's default backend choice.

Combining compilation with one KV head gives a second combined speedup: the
same workload measured 2.51 ms eager and 1.45 ms compiled for GQA-1, or 2.46x
relative to the original 3.58 ms MHA kernel. GQA-2 remains the safer quality/
capacity tradeoff for training (2.57 ms eager, 1.47 ms compiled).

GQA reduces the QKV projection width and is the best direct candidate for this
path. The VHA comparison follows the [Virtual Head Attention paper]
(https://arxiv.org/abs/2610.17959) and the Apache-2.0
[PaddlePaddle implementation at commit `ab7ceded`](https://github.com/PaddlePaddle/VHA/tree/ab7ceded6e1ad6a6ba635afd19a152bc64fe8e2d).
VHA has lower physical QKV width, but its Premix and Postmix must be fused or
folded into projections; naive runtime mixing can erase the gain. Neither
changes the virtual-head SDPA interaction count, so they cannot provide a 2x
end-to-end training speedup by themselves.
On the packed attention workload, the current GB300 measurements were 3.58 ms
for MHA, 2.58 ms for GQA-2, and 4.52 ms for runtime VHA. VHA without the
runtime Postmix was 2.96 ms; this is a useful lower bound, not a train-time
configuration. The paper's Premix/Postmix folding is therefore the appropriate
inference optimization, while GQA is the useful train-time choice.
The model accepts `model.n_kv_head`; leave it `null` for the original MHA
checkpoint layout, or set (for example) `2` when training a new GQA model.

On the full B8/S4096 603.1M GQA-2 model, the current grouped path measures
**255.1 ms** per forward/backward step. Inductor's `reduce-overhead` mode was
also tested end to end and regressed to about **1.71 s** because the dynamic
routed backward graph does not reach the CUDA-graph fast path; leave the
default compile mode enabled for training.

For fixed-shape single-document batches, setting `LOL_COMPILE_DYNAMIC=0` lets
Inductor specialize the routed dispatch dimensions. One clean run measured
**250.5 ms** at B8/S4096 (about 1.8% faster), but this path is sensitive to
host load and remains opt-in until repeated runs confirm the gain.

The routed profile confirms why further attention-only work has limited
end-to-end impact: in a profiled B1/S4096 step, FlashAttention backward and
its transpose kernels account for about **7.8 ms** of CUDA time, while grouped
GEMMs and routing/elementwise kernels account for the majority of the step.
Use `--profile` on `benchmarks/lol_step_baseline.py` to reproduce the table.

The benchmark also exposes a training tradeoff that keeps the full forward
receptive field while limiting autograd to the last recurrent depths:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/lol_step_baseline.py \
  --mode grouped --batch 8 --sequence-length 4096 \
  --backprop-steps 4 --warmup 5 --steps 10
```

On the GB300, a clean B8/S4096 run took 184.5 ms with only the final four
depths differentiated, versus 272.7 ms for full-depth grouped MHA (1.48x).
This changes the gradient approximation and should be treated as a training
schedule choice. Executing only eight of the 24 recurrent depths gives 129.4
ms (2.11x), but changes the forward depth and therefore the model computation.
Use `--num-steps 8` only when that tradeoff is intended.
With fair GQA-2 (603.1M parameters), the same eight-depth schedule measured
123.0 ms at B8/S4096, or **2.22x** versus full-depth grouped MHA.

Grouped dispatch now casts each expert projection to the BF16 compute dtype
before stacking and transposing it. It also applies the router residual weight
while outputs remain in expert-sorted order, avoiding a restored `[B,T,D]`
residual pass. These changes preserve autograd links and reduced the full
grouped step by about 4–6% in matched B8 runs; isolated weight packing fell
about 8%.

## Plotting
Copy [the manifest template](plotting/assets/datasets.example.json) to `datasets.json` in your data directory, replace null placeholders, and provide the files required by your selected plot. The default directory is `plotting/assets`; select another with `--data-dir`. Relative input paths must stay inside that directory.
