# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import re

from vllm.config import get_current_vllm_config

LAYER_ADAPTIVE_TAIL_LAYERS = 8
_LAYER_INDEX_RE = re.compile(r"\.(\d+)\.")


def extract_layer_index_from_name(layer_name: str) -> int:
    match = _LAYER_INDEX_RE.search(layer_name)
    if match is None:
        raise ValueError(f"Unable to extract layer index from layer name: {layer_name}")
    return int(match.group(1))


def should_use_layer_adaptive_dense_cache(layer_name: str | None) -> bool:
    if not layer_name:
        return False
    try:
        vllm_config = get_current_vllm_config()
    except AssertionError:
        return False
    if not vllm_config.attention_config.turboquant_layer_adaptive:
        return False
    total_layers = getattr(vllm_config.model_config.hf_text_config, "num_hidden_layers", 0)
    if total_layers <= 0:
        return False
    layer_idx = extract_layer_index_from_name(layer_name)
    return layer_idx >= max(total_layers - LAYER_ADAPTIVE_TAIL_LAYERS, 0)
