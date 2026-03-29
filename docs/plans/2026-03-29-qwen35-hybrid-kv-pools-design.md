# Qwen3.5 Hybrid KV Pools Design

## Problem

`Qwen3.5` hybrid multimodal models combine:

- full-attention layers, where TurboQuant can compress the KV cache, and
- linear-attention (`mamba`/`gdn`) layers, which use a different state layout.

The current vLLM hybrid allocator assumes one shared block-pool geometry across
all hybrid groups. That is stable, but it forces TurboQuant full-attention
layers to live inside a page shape sized for the larger hybrid contract rather
than their true packed storage.

## What We Tried

We prototyped a direct split between:

- logical hybrid page size for grouping/scheduling, and
- physical packed page size for worker allocation.

That made the worker/backend side cleaner, but it broke the single-pool memory
budget assumptions. In practice, the engine started budgeting packed
full-attention pages and linear-attention state pages as separate physical pools
without a matching scheduler/block-pool change.

## Options

### 1. Keep single-pool padding

Pros:

- stable today
- no scheduler refactor
- current `turbo3` and `turbo4` smoke remain green

Cons:

- leaves hybrid memory savings on the table
- not enough for a strong "real packed KV allocator" claim

### 2. Logical/physical split only

Pros:

- small worker/backend patch set
- attractive on paper

Cons:

- not enough for Qwen3.5 hybrid
- violates current shared-pool accounting assumptions
- regressed memory budgeting during engine startup

### 3. Heterogeneous hybrid block pools

Pros:

- correct long-term architecture
- enables real TurboQuant wins for full-attention groups
- keeps linear-attention state sizing honest

Cons:

- requires scheduler and block-pool changes
- connector/offload code will need a follow-up pass

## Recommendation

Implement option 3 and keep option 1 as the stable baseline until it lands.

That means:

1. leave the current backend and smoke path intact,
2. introduce heterogeneous block-pool metadata in the KV config,
3. teach the scheduler/block manager to allocate blocks per physical pool
   rather than assuming one shared hybrid geometry,
4. update worker reshape/bind logic only after the new pool contract exists.

## Likely Change Surface

- `vllm/v1/core/kv_cache_utils.py`
  - represent heterogeneous physical pool blueprints in `KVCacheConfig`
- `vllm/v1/core/block_pool.py`
  - support multiple physical pools or pool classes
- `vllm/v1/core/single_type_kv_cache_manager.py`
  - track block availability per hybrid physical pool
- `vllm/v1/worker/gpu_model_runner.py`
  - allocate and bind per-pool tensors
- `vllm/v1/worker/gpu/attn_utils.py`
  - reshape per-pool KV tensors
- connector/offload code
  - follow the new pool contract after scheduler/worker are stable

## Success Criteria

- `turbo4` remains the default Qwen3.5 hybrid mode
- `turbo3` remains available as the higher-compression mode
- multimodal smoke stays green
- packed full-attention KV memory is reflected in allocator math, not just
  inside the backend tensor payload
- benchmark tables show a real memory delta versus the current padded baseline
