# Qwen3.5 Hybrid TurboQuant Status

## What Works

- `QWEN_HYBRID_TURBOQUANT` is wired through backend selection for multimodal
  `Qwen3.5` hybrid models.
- Both `turbo4` and `turbo3` pass the dummy-engine smoke path with
  `Qwen/Qwen3.5-4B`.
- The backend uses packed TurboQuant K/V storage logic and dequant-on-read.
- Runtime flags are exposed through `AttentionConfig` and CLI.
- Worker-side allocation now distinguishes logical hybrid page size from the
  physical packed TurboQuant page size.

## Current Limitation

The current design still keeps the hybrid scheduler and coordinator on the
existing single-pool semantics. Logical page sizes remain padded where hybrid
grouping needs them, while worker allocation and reshape use packed physical
page sizes.

This means:

- backend selection and execution work end-to-end,
- `turbo3` and `turbo4` are both runnable,
- packed allocation is active in the worker path,
- but connector/offload coverage and broader benchmark validation are still
  incomplete.

## What We Verified Next

We validated a direct split between:

- logical hybrid page size used by the scheduler/grouping code, and
- physical packed page size used by raw tensor allocation.

The working version uses the physical page size for KV tensor sizing and worker
reshape while preserving the logical page size for the hybrid grouping logic.
That keeps the current `BlockPool` contract intact and restores stable smoke
execution for both modes.

## Correct Next Design

The next milestone is broader system validation rather than another backend-side
packing tweak:

1. Keep the current backend path as the stable working PoC.
2. Validate connector and offload code paths against the physical page-size
   split.
3. Add deeper measurements for memory, throughput, and multimodal prompts.
4. Decide whether a later multi-pool hybrid coordinator is still worth the
   added complexity.

## Publishable State

This branch is already strong enough to show:

- a new `vLLM` attention backend,
- hybrid-model detection for `Qwen3.5`,
- dual TurboQuant mode support,
- successful end-to-end smoke execution through the v1 engine.

## Remaining Engineering Work

1. Validate connector/offload paths against the physical page-size split.
2. Add real 27B multimodal launch validation and benchmark tables.
3. Extend beyond text smoke into image-bearing prompts and service-level tests.
4. Revisit a multi-pool hybrid coordinator only if the current split leaves
   measurable headroom on long-context workloads.
