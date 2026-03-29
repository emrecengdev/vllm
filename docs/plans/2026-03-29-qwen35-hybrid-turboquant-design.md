# Qwen3.5 Hybrid TurboQuant Plan

This fork targets a production-quality TurboQuant integration for Qwen3.5
hybrid multimodal models on vLLM.

## Scope

- Runtime: `vLLM`
- Models: `Qwen3.5` multimodal hybrid variants
- Modes:
  - `turbo4` as default balanced mode
  - `turbo3` as optional aggressive mode
- Future features:
  - Sparse-V decode optimization
  - layer-adaptive precision
  - asymmetric K/V policies

## Integration Strategy

1. Add a dedicated attention backend surface instead of mutating the existing
   flash attention backend in-place.
2. Keep phase-0 behavior logically equivalent to FlashAttention so that backend
   selection, CLI flags, and hybrid model routing can be validated first.
3. In phase-1, introduce packed TurboQuant KV storage for full-attention layers.
4. Preserve linear-attention states in their native format.
5. Add Sparse-V and adaptive policies only after correctness and memory
   accounting are stable.

## Planned Code Areas

- `vllm/config/attention.py`
- `vllm/engine/arg_utils.py`
- `vllm/v1/attention/backends/qwen_hybrid_turboquant.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/model_executor/models/qwen3_5.py`

## Phase Breakdown

### Phase 0

- dedicated backend enum and file
- CLI/config flags
- Qwen3.5 hybrid multimodal detection helper
- no packed KV yet

### Phase 1

- turbo4 packed KV format
- full-attention-only quantized storage
- backend-side KV update path
- metrics for compressed vs baseline bytes

### Phase 2

- turbo3 mode
- Sparse-V decode path
- layer-adaptive precision

## Success Criteria

- Qwen3.5 multimodal model loads with the dedicated backend
- backend is selectable from CLI
- no regression to existing backends
- later phases add measurable KV memory reduction on long contexts
