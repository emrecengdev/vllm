# Qwen3.5 Hybrid TurboQuant Status

## What Works

- `QWEN_HYBRID_TURBOQUANT` is wired through backend selection for multimodal
  `Qwen3.5` hybrid models.
- `turbo4` and `turbo3` both pass text smoke with `Qwen/Qwen3.5-4B`.
- `turbo4` and `turbo3` both pass multimodal smoke with an image-bearing prompt
  against `Qwen/Qwen3.5-4B`.
- The backend uses packed TurboQuant K/V storage logic and dequant-on-read.
- `Sparse V` is wired for decode-time TurboQuant value dequant skipping.
- Layer-adaptive fallback is wired so the tail layers can keep dense
  `FullAttentionSpec` KV layout while the rest of the hybrid stack remains
  TurboQuant-packed.
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

- `pytest tests/v1/attention/test_qwen_hybrid_turboquant.py tests/v1/core/test_kv_cache_utils.py`
  - `56 passed`
- text smoke
  - `turbo4`: `llm_init_ok`, `generate_ok`
  - `turbo3`: `llm_init_ok`, `generate_ok`
- multimodal smoke
  - `turbo4`: `llm_init_ok`, `generate_ok`
  - `turbo3`: `llm_init_ok`, `generate_ok`
- text smoke with advanced policies enabled
  - `turbo4 + sparse_v + layer_adaptive`: `llm_init_ok`, `generate_ok`
  - `turbo3 + sparse_v + layer_adaptive`: `llm_init_ok`, `generate_ok`
- multimodal smoke with advanced policies enabled
  - `turbo4 + sparse_v + layer_adaptive`: `llm_init_ok`, `generate_ok`
  - `turbo3 + sparse_v + layer_adaptive`: `llm_init_ok`, `generate_ok`
- observed KV capacity deltas after physical packed allocation landed
  - `turbo4` text smoke at `GPU_MEMORY_UTILIZATION=0.6`
    - before: `24,064` tokens
    - after: `27,136` tokens
  - `turbo3` text smoke at `GPU_MEMORY_UTILIZATION=0.5`
    - before: `8,192` tokens
    - after: `10,240` tokens
- real single-`3090` `27B` validation with `Kbenkhaled/Qwen3.5-27B-NVFP4`
  - text smoke
    - `turbo4 + sparse_v + layer_adaptive`
    - `GPU_MEMORY_UTILIZATION=0.90`
    - `max_model_len=512`
    - `llm_init_ok`, `generate_ok`
  - multimodal smoke
    - `turbo4 + sparse_v + layer_adaptive`
    - `TURBOQUANT_GPU_MEMORY_UTILIZATION=0.90`
    - `max_model_len=512`
    - `llm_init_ok`, `generate_ok`
  - branch-local compatibility fixes required for this path
    - compressed-tensors config sanitization for `scale_dtype` / `zp_dtype`
    - fallback for older `QuantizationStrategy` enums without `ATTN_HEAD`
    - conservative FLA/GDN Triton autotune on Ampere
    - process-global GDN prefill warmup deduplication to avoid per-layer
      startup amplification

## Current Limitation

The branch now carries the packed physical layout end-to-end for the main
TurboQuant worker path, including `Sparse V` and layer-adaptive fallback, but
two areas are still not fully hardened:

- connector/offload paths have not yet been revalidated against the new
  physical pool contract,
- the public benchmark story still needs a cleaner `27B` benchmark pack and
  OpenAI-server numbers beyond smoke coverage.

## Publishable State

This branch is already strong enough to show:

- a new `vLLM` attention backend for `Qwen3.5` hybrid models,
- dual `turbo4` and `turbo3` support,
- `Sparse V` and layer-adaptive TurboQuant policies,
- text and multimodal end-to-end smoke execution,
- heterogeneous physical-pool metadata and scheduler plumbing in the v1 KV
  cache path,
- real packed-page KV capacity gains in the worker allocation path.

## Remaining Engineering Work

1. Revalidate connector/offload paths against the physical pool contract.
2. Add benchmark tables for the validated single-`3090` `27B NVFP4` recipe.
3. Benchmark OpenAI-server serving defaults on the 24 GB target recipe.
