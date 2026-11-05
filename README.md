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
