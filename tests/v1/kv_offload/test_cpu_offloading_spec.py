# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
import pytest

from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


def make_vllm_config(cpu_bytes_to_use: int):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector_extra_config={"cpu_bytes_to_use": cpu_bytes_to_use}
        ),
        cache_config=SimpleNamespace(block_size=16),
        parallel_config=SimpleNamespace(world_size=2),
        kv_events_config=None,
    )


def make_full_attn_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=64,
        head_size_v=64,
        dtype=torch.float16,
    )


def test_cpu_offloading_spec_uses_tensor_bytes_per_block():
    kv_cache_config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(size=1000, num_blocks=10, shared_by=["layer0"]),
            KVCacheTensor(size=2000, num_blocks=10, shared_by=["layer1"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer0", "layer1"],
                kv_cache_spec=make_full_attn_spec(),
            )
        ],
    )

    spec = CPUOffloadingSpec(
        make_vllm_config(cpu_bytes_to_use=6000),
        kv_cache_config,
    )

    assert spec.num_blocks == 10


def test_cpu_offloading_spec_rejects_multiple_kv_groups():
    kv_cache_config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(size=1000, num_blocks=10, shared_by=["layer0"]),
            KVCacheTensor(size=1000, num_blocks=10, shared_by=["layer1"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer0"],
                kv_cache_spec=make_full_attn_spec(),
            ),
            KVCacheGroupSpec(
                layer_names=["layer1"],
                kv_cache_spec=make_full_attn_spec(),
            ),
        ],
    )

    with pytest.raises(NotImplementedError, match="single KV cache group"):
        CPUOffloadingSpec(
            make_vllm_config(cpu_bytes_to_use=6000),
            kv_cache_config,
        )
