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

## Current Limitation

The branch now has heterogeneous-pool scheduling metadata, but worker-side raw
tensor allocation still follows the logical hybrid page geometry.

Practically, that means:

- the backend and hybrid scheduler path are stable,
- text and multimodal execution both work for `turbo4` and `turbo3`,
- per-pool block accounting exists in the core allocator path,
- but the final worker allocation path does not yet allocate raw KV tensors
  using the packed physical page size.

So this branch is stronger than the original single-pool PoC, but it is still
not the final "full packed memory win" milestone.

## Publishable State

This branch is already strong enough to show:

- a new `vLLM` attention backend for `Qwen3.5` hybrid models,
- dual `turbo4` and `turbo3` support,
- text and multimodal end-to-end smoke execution,
- heterogeneous physical-pool metadata and scheduler plumbing in the v1 KV
  cache path.

## Remaining Engineering Work

1. Teach worker-side raw KV tensor allocation and reshape paths to use
   physical packed page sizes instead of only logical padded page sizes.
2. Revalidate connector/offload paths once worker allocation is pool-aware.
3. Add real `27B` multimodal launch validation and benchmark tables.
4. Measure true memory deltas once packed physical allocation lands.
