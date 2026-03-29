import torch
from types import SimpleNamespace

from vllm.v1.attention.backends.qwen_hybrid_turboquant import (
    QwenHybridTurboQuantBackend,
    is_qwen_hybrid_turboquant_candidate,
)
from vllm.v1.attention.backends.qwen_hybrid_turboquant_utils import (
    should_use_layer_adaptive_dense_cache,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, TurboQuantFullAttentionSpec


def test_turboquant_full_attention_spec_sizes():
    turbo4 = TurboQuantFullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
        mode="turbo4",
    )
    turbo3 = TurboQuantFullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
        mode="turbo3",
    )

    assert turbo4.bits == 4
    assert turbo4.packed_k_bytes == 64
    assert turbo4.packed_v_bytes == 64
    assert turbo4.bytes_per_token_per_head == 132
    assert turbo4.real_page_size_bytes == 16 * 8 * 132

    assert turbo3.bits == 3
    assert turbo3.packed_k_bytes == 48
    assert turbo3.packed_v_bytes == 48
    assert turbo3.bytes_per_token_per_head == 100
    assert turbo3.real_page_size_bytes == 16 * 8 * 100


def test_turboquant_full_attention_copy_scales_padding():
    spec = TurboQuantFullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
        mode="turbo4",
        page_size_padded=16384,
    )

    resized = spec.copy_with_new_block_size(32)
    assert resized.block_size == 32
    assert resized.page_size_padded == 32768


def test_qwen_hybrid_turboquant_candidate_detection():
    candidate = SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal_model=True,
            hf_config=SimpleNamespace(model_type="qwen3_5"),
            hf_text_config=SimpleNamespace(
                layer_types=[
                    "linear_attention",
                    "full_attention",
                    "linear_attention",
                ]
            ),
        )
    )
    non_candidate = SimpleNamespace(
        model_config=SimpleNamespace(
            is_multimodal_model=False,
            hf_config=SimpleNamespace(model_type="qwen3_5"),
            hf_text_config=SimpleNamespace(
                layer_types=["full_attention", "full_attention"]
            ),
        )
    )

    assert is_qwen_hybrid_turboquant_candidate(candidate) is True
    assert is_qwen_hybrid_turboquant_candidate(non_candidate) is False


def test_qwen_hybrid_turboquant_backend_shape_fallback():
    shape = QwenHybridTurboQuantBackend.get_kv_cache_shape(
        num_blocks=7,
        block_size=16,
        num_kv_heads=8,
        head_size=128,
    )

    assert shape[0] == 7
    assert shape[1] == 16
    assert shape[2] == 8
    assert shape[3] == 132


def test_qwen_hybrid_turboquant_backend_shape_for_spec():
    turbo_spec = TurboQuantFullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
        mode="turbo4",
    )
    dense_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        dtype=torch.float16,
    )

    turbo_shape = QwenHybridTurboQuantBackend.get_kv_cache_shape_for_spec(7, turbo_spec)
    dense_shape = QwenHybridTurboQuantBackend.get_kv_cache_shape_for_spec(7, dense_spec)

    assert turbo_shape == (7, 16, 8, 132)
    assert dense_shape == (2, 7, 16, 8, 128)


def test_layer_adaptive_dense_cache_helper(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.attention.backends.qwen_hybrid_turboquant_utils.get_current_vllm_config",
        lambda: SimpleNamespace(
            attention_config=SimpleNamespace(turboquant_layer_adaptive=True),
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(num_hidden_layers=64)
            ),
        ),
    )

    assert should_use_layer_adaptive_dense_cache("model.layers.56.self_attn") is True
    assert should_use_layer_adaptive_dense_cache("model.layers.10.self_attn") is False
