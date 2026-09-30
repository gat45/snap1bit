# snap1bit

Experimental Q1_0 execution support for **Hexagon HTP v81** in a
`ggml-hexagon` / llama.cpp fork.

This repository documents a runtime experiment. It does **not** claim that
post-training quantizing an arbitrary LLM to one bit preserves model quality.

## Why it is useful

The work separates two questions that are often conflated:

1. Can the HTP execute the Q1_0 representation correctly and efficiently?
2. Does a particular model remain useful after naive Q1_0 post-training
   quantization?

Stage 1 answers the first question positively. Q1_0 weights are repacked
losslessly into the existing Q4_0 tiled execution layout, so the backend can
reuse the validated HTP Q4_0 matrix-vector path rather than falling back to
the CPU. This is a compatibility and execution-path improvement; it is not
yet a compact native Q1_0 DSP kernel.

## Stage 1 status

Tested configuration:

- Snapdragon SM8850 / Hexagon HTP v81
- fork based on `self-build-jz` commit `37f752b` (upstream b11855)
- Stage 1 repack patches applied
- `test-backend-ops -b HTP0`: 32/32 Q1_0 `MUL_MAT` checks passed
- full backend test suite: 796/796 passed

On Nanbeige 4.2-3B Q1_0, HTP0 measured **15.42 tokens/s** decode (`tg128`),
versus **6.05 tokens/s** on the CPU reference path. The result demonstrates
that the HTP execution route is engaged and substantially faster than CPU for
this representation.

## Critical quality limitation

The same Nanbeige model, quantized naively to Q1_0, is not a usable language
model: a controlled CPU/HTP perplexity matrix found approximately `2.1e8` PPL
on both paths, while the Q4_K_M control was around `40` under the same short
WikiText-2 protocol. The agreement between CPU and HTP identifies naive
Q1_0 PTQ as the failure mode, not an HTP correctness failure.

Accordingly, **Q1_0 must be opt-in** and used only for tensors with measured
quality evidence in a mixed-precision plan. A fast kernel cannot recover
information discarded by an unsuitable quantizer.

This distinction also matters for BitNet-style models: BitNet b1.58 uses
training-time ternary weights (`{-1, 0, +1}`), which are not the same format
or claim as GGML Q1_0. See [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764).

## What Stage 1 is and is not

| Property | Stage 1 |
|---|---|
| HTP Q1_0 support | Yes, through repack to Q4_0 tiled execution layout |
| Dedicated DSP Q1_0 symbol/kernel | No |
| Extra model-quality guarantee | No |
| Safe default for arbitrary PTQ models | No |
| CPU fallback avoidance for supported Q1_0 tensors | Yes |

## Next step: Stage 2

Stage 2 targets a native compact Q1_0 tile (192-byte weight tile) to reduce
weight traffic instead of expanding into the Q4_0 execution representation.
Before treating it as an optimization, it needs:

1. bit-exact tile/repack self-tests;
2. backend correctness tests over production matrix shapes;
3. HTP-versus-CPU logits and perplexity checks;
4. bandwidth and HTP occupancy measurements; and
5. mixed-precision, per-tensor quality measurements.

## Reproducibility requirements

Always record the runtime commit, Stage 1/Stage 2 kernel variant, DSP skel
hash, backend mode (`dspqueue` or `mempool`), device, model hash, thermal
state, and exact benchmark command. `b11855` alone is not a sufficient build
identity: stock b11855 and b11855 plus Stage 1 do not expose the same Q1_0
capability.
