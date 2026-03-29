# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental TurboQuant backend for Qwen3.5 hybrid models.

This backend is intentionally conservative in the first iteration:
- It inherits the FlashAttention execution path.
- It exists to provide a stable integration point for hybrid-aware KV cache
  compression without modifying existing backends in-place.
- Packed TurboQuant KV storage and Sparse-V decode paths will be added
  incrementally on top of this class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger

from .flash_attn import FlashAttentionBackend, FlashAttentionImpl

logger = init_logger(__name__)


@dataclass
class TurboQuantRuntimeConfig:
    enabled: bool
    mode: str
    sparse_v: bool
    layer_adaptive: bool
    debug_log_stats: bool

    @classmethod
    def from_current_config(cls) -> "TurboQuantRuntimeConfig":
        config = get_current_vllm_config().attention_config
        return cls(
            enabled=config.turboquant_enabled,
            mode=config.turboquant_mode,
            sparse_v=config.turboquant_sparse_v,
            layer_adaptive=config.turboquant_layer_adaptive,
            debug_log_stats=config.turboquant_debug_log_stats,
        )


def is_qwen_hybrid_turboquant_candidate(vllm_config: Any) -> bool:
    model_config = vllm_config.model_config
    hf_text_config = getattr(model_config, "hf_text_config", None)
    layer_types = getattr(hf_text_config, "layer_types", None)
    if not layer_types:
        return False
    has_full_attention = any(layer_type == "full_attention" for layer_type in layer_types)
    has_linear_attention = any(layer_type == "linear_attention" for layer_type in layer_types)
    return bool(
        model_config.is_multimodal_model
        and getattr(model_config.hf_config, "model_type", None) in {"qwen3_5", "qwen3_5_moe"}
        and has_full_attention
        and has_linear_attention
    )


class QwenHybridTurboQuantBackend(FlashAttentionBackend):
    """FlashAttention-derived integration point for TurboQuant on Qwen3.5.

    The first implementation intentionally keeps the same logical KV cache
    shape as FlashAttention. This allows us to:
    1. plumb runtime flags through the stack,
    2. validate backend selection and hybrid-model detection,
    3. add packed KV storage in a follow-up patch with contained blast radius.
    """

    @staticmethod
    def get_name() -> str:
        return "QWEN_HYBRID_TURBOQUANT"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return QwenHybridTurboQuantImpl

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True


class QwenHybridTurboQuantImpl(FlashAttentionImpl):
    """Phase-0 implementation.

    This currently reuses FlashAttention's forward path unchanged while exposing
    a dedicated backend name and runtime config surface. The packed TurboQuant KV
    path will eventually override KV update and cache layout handling here.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.turboquant_config = TurboQuantRuntimeConfig.from_current_config()
        if self.turboquant_config.debug_log_stats:
            logger.info(
                "Initialized %s with mode=%s sparse_v=%s layer_adaptive=%s",
                self.__class__.__name__,
                self.turboquant_config.mode,
                self.turboquant_config.sparse_v,
                self.turboquant_config.layer_adaptive,
            )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().forward(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )
