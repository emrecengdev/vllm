# Qwen3.5 Hybrid TurboQuant Status

## What Works

- `QWEN_HYBRID_TURBOQUANT` is wired through backend selection for multimodal
  `Qwen3.5` hybrid models.
- `turbo4` and `turbo3` both pass text smoke with `Qwen/Qwen3.5-4B`.
- `turbo4` and `turbo3` both pass multimodal smoke with an image-bearing prompt
  against `Qwen/Qwen3.5-4B`.
- The backend uses packed TurboQuant K/V storage logic and dequant-on-read.
- Runtime flags are exposed through `AttentionConfig` and CLI.
- KV cache config/coordinator/manager now carry heterogeneous physical-pool
  metadata:
  - `KVCacheConfig.num_blocks_by_pool`
  - `KVCacheGroupSpec.physical_pool_id`
  - `KVCacheTensor.physical_pool_id`
- The v1 scheduler admission path now checks capacity per physical pool instead
  of assuming one global block pool.
- Worker-side KV tensor allocation and reshape now use physical packed page
  sizes for the TurboQuant pool instead of the old logical padded size.

## What We Verified

Validation completed on 30 March 2026:

- `pytest tests/v1/attention/test_qwen_hybrid_turboquant.py`
  - `4 passed`
- `pytest tests/v1/attention/test_qwen_hybrid_turboquant.py tests/v1/core/test_kv_cache_utils.py`
  - `54 passed`
- text smoke
  - `turbo4`: `llm_init_ok`, `generate_ok`
  - `turbo3`: `llm_init_ok`, `generate_ok`
- multimodal smoke
  - `turbo4`: `llm_init_ok`, `generate_ok`
  - `turbo3`: `llm_init_ok`, `generate_ok`
- observed KV capacity deltas after physical packed allocation landed
  - `turbo4` text smoke at `GPU_MEMORY_UTILIZATION=0.6`
    - before: `24,064` tokens
    - after: `27,136` tokens
  - `turbo3` text smoke at `GPU_MEMORY_UTILIZATION=0.5`
    - before: `8,192` tokens
    - after: `10,240` tokens

## Current Limitation

The branch now carries the packed physical layout end-to-end for the main
TurboQuant worker path, but two areas are still not fully hardened:

Practically, that means:

- the backend and hybrid scheduler path are stable,
- text and multimodal execution both work for `turbo4` and `turbo3`,
- the main worker path now realizes real KV capacity gains from physical packed
  pages,
- but connector/offload paths have not yet been revalidated against the new
  physical pool contract.

So this branch is past the original PoC stage and now demonstrates actual KV
memory savings, but it is not yet the final production-hardening milestone.

## Publishable State

This branch is already strong enough to show:

- a new `vLLM` attention backend for `Qwen3.5` hybrid models,
- dual `turbo4` and `turbo3` support,
- text and multimodal end-to-end smoke execution,
- heterogeneous physical-pool metadata and scheduler plumbing in the v1 KV
  cache path,
- real packed-page KV capacity gains in the worker allocation path.

## Remaining Engineering Work

1. Revalidate connector/offload paths against the physical pool contract.
2. Add real `27B` multimodal launch validation and benchmark tables.
3. Benchmark OpenAI-server serving defaults on the 24 GB target recipe.
4. Layer in `Sparse V` and higher-level TurboQuant policies.
