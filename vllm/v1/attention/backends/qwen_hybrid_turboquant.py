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
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec, TurboQuantFullAttentionSpec

from .flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
    cascade_attention,
    flash_attn_varlen_func,
)
from .qwen_hybrid_turboquant_utils import should_use_layer_adaptive_dense_cache

logger = init_logger(__name__)
SPARSE_V_ALPHA_THRESHOLD = 1e-6


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

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        mode = "turbo4"
        try:
            vllm_config = get_current_vllm_config()
            mode = vllm_config.attention_config.turboquant_mode
        except AssertionError:
            # Some static shape queries run before vLLM config context exists.
            pass
        spec = TurboQuantFullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            head_size_v=head_size,
            dtype=torch.uint8,
            mode=mode,
        )
        return (
            num_blocks,
            block_size,
            num_kv_heads,
            spec.bytes_per_token_per_head,
        )

    @classmethod
    def get_kv_cache_shape_for_spec(
        cls,
        num_blocks: int,
        kv_cache_spec,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if isinstance(kv_cache_spec, TurboQuantFullAttentionSpec):
            return cls.get_kv_cache_shape(
                num_blocks,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
                cache_dtype_str,
            )
        if isinstance(kv_cache_spec, FullAttentionSpec):
            return FlashAttentionBackend.get_kv_cache_shape(
                num_blocks,
                kv_cache_spec.block_size,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
                cache_dtype_str,
            )
        raise TypeError(f"Unsupported KV cache spec for TurboQuant backend: {type(kv_cache_spec)}")

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4)
        return (0, 1, 2, 3)

    @classmethod
    def validate_configuration(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        use_per_head_quant_scales: bool,
        device_capability,
        attn_type: str,
    ) -> list[str]:
        invalid_reasons = super().validate_configuration(
            head_size=head_size,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            block_size=block_size,
            use_mla=use_mla,
            has_sink=has_sink,
            use_sparse=use_sparse,
            use_mm_prefix=use_mm_prefix,
            use_per_head_quant_scales=use_per_head_quant_scales,
            device_capability=device_capability,
            attn_type=attn_type,
        )
        vllm_config = get_current_vllm_config()
        if not TurboQuantRuntimeConfig.from_current_config().enabled:
            invalid_reasons.append("turboquant_enabled is false")
        if not is_qwen_hybrid_turboquant_candidate(vllm_config):
            invalid_reasons.append("model is not a supported Qwen3.5 hybrid multimodal candidate")
        return invalid_reasons


class QwenHybridTurboQuantImpl(FlashAttentionImpl):
    """Phase-0 implementation.

    This currently reuses FlashAttention's forward path unchanged while exposing
    a dedicated backend name and runtime config surface. The packed TurboQuant KV
    path will eventually override KV update and cache layout handling here.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=sinks,
        )
        self.turboquant_config = TurboQuantRuntimeConfig.from_current_config()
        self.bits = 4 if self.turboquant_config.mode == "turbo4" else 3
        self.scale_dtype = torch.float16
        self.headdim = head_size
        self.head_size_v = head_size
        if self.turboquant_config.debug_log_stats:
            logger.info(
                "Initialized %s with mode=%s sparse_v=%s layer_adaptive=%s",
                self.__class__.__name__,
                self.turboquant_config.mode,
                self.turboquant_config.sparse_v,
                self.turboquant_config.layer_adaptive,
            )

    def _uses_dense_layer_fallback(self, layer: torch.nn.Module) -> bool:
        return should_use_layer_adaptive_dense_cache(
            getattr(layer, "layer_name", None)
        )

    @property
    def _quant_min(self) -> int:
        return -(1 << (self.bits - 1))

    @property
    def _quant_max(self) -> int:
        return (1 << (self.bits - 1)) - 1

    @property
    def _unsigned_max(self) -> int:
        return (1 << self.bits) - 1

    @property
    def packed_k_bytes(self) -> int:
        return TurboQuantFullAttentionSpec.packed_bytes(self.headdim, self.bits)

    @property
    def packed_v_bytes(self) -> int:
        return TurboQuantFullAttentionSpec.packed_bytes(self.head_size_v, self.bits)

    @property
    def scale_bytes(self) -> int:
        return torch.tensor([], dtype=self.scale_dtype).element_size()

    @property
    def bytes_per_token_per_head(self) -> int:
        return self.packed_k_bytes + self.packed_v_bytes + 2 * self.scale_bytes

    @property
    def _value_start(self) -> int:
        return self.packed_k_bytes

    @property
    def _k_scale_start(self) -> int:
        return self.packed_k_bytes + self.packed_v_bytes

    @property
    def _v_scale_start(self) -> int:
        return self._k_scale_start + self.scale_bytes

    def _pad_last_dim(self, tensor: torch.Tensor, multiple: int) -> torch.Tensor:
        pad = (-tensor.shape[-1]) % multiple
        if pad:
            tensor = F.pad(tensor, (0, pad))
        return tensor

    def _pack_unsigned(self, values: torch.Tensor) -> torch.Tensor:
        values = values.to(torch.int32)
        if self.bits == 4:
            values = self._pad_last_dim(values, 2)
            values = values.view(*values.shape[:-1], -1, 2)
            packed = values[..., 0] | (values[..., 1] << 4)
            return packed.to(torch.uint8)

        values = self._pad_last_dim(values, 8)
        values = values.view(*values.shape[:-1], -1, 8)
        b0 = values[..., 0] | (values[..., 1] << 3) | ((values[..., 2] & 0x3) << 6)
        b1 = (
            ((values[..., 2] >> 2) & 0x1)
            | (values[..., 3] << 1)
            | (values[..., 4] << 4)
            | ((values[..., 5] & 0x1) << 7)
        )
        b2 = ((values[..., 5] >> 1) & 0x3) | (values[..., 6] << 2) | (values[..., 7] << 5)
        packed = torch.stack((b0, b1, b2), dim=-1)
        return packed.reshape(*packed.shape[:-2], -1).to(torch.uint8)

    def _unpack_unsigned(
        self,
        packed: torch.Tensor,
        num_values: int,
    ) -> torch.Tensor:
        packed = packed.to(torch.int32)
        if self.bits == 4:
            lo = packed & 0x0F
            hi = (packed >> 4) & 0x0F
            unpacked = torch.stack((lo, hi), dim=-1).reshape(*packed.shape[:-1], -1)
            return unpacked[..., :num_values]

        packed = packed.view(*packed.shape[:-1], -1, 3)
        b0 = packed[..., 0]
        b1 = packed[..., 1]
        b2 = packed[..., 2]
        unpacked = torch.stack(
            (
                b0 & 0x7,
                (b0 >> 3) & 0x7,
                ((b0 >> 6) & 0x3) | ((b1 & 0x1) << 2),
                (b1 >> 1) & 0x7,
                (b1 >> 4) & 0x7,
                ((b1 >> 7) & 0x1) | ((b2 & 0x3) << 1),
                (b2 >> 2) & 0x7,
                (b2 >> 5) & 0x7,
            ),
            dim=-1,
        ).reshape(*packed.shape[:-2], -1)
        return unpacked[..., :num_values]

    def _quantize_tensor(
        self,
        tensor: torch.Tensor,
        num_values: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_abs = tensor.abs().amax(dim=-1, keepdim=True)
        scale = torch.where(
            max_abs > 0,
            max_abs / float(max(self._quant_max, 1)),
            torch.ones_like(max_abs),
        )
        quantized = torch.round(tensor / scale).clamp(self._quant_min, self._quant_max)
        unsigned = (quantized.to(torch.int32) - self._quant_min).to(torch.uint8)
        packed = self._pack_unsigned(unsigned)
        return packed[..., : TurboQuantFullAttentionSpec.packed_bytes(num_values, self.bits)], scale

    def _encode_scale_bytes(self, scale: torch.Tensor) -> torch.Tensor:
        return scale.to(self.scale_dtype).contiguous().view(torch.uint8)

    def _decode_scale_bytes(self, scale_bytes: torch.Tensor) -> torch.Tensor:
        return scale_bytes.contiguous().view(self.scale_dtype)

    def _dequantize_tensor(
        self,
        packed: torch.Tensor,
        scale_bytes: torch.Tensor,
        num_values: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        scale = self._decode_scale_bytes(scale_bytes).to(dtype)
        unsigned = self._unpack_unsigned(packed, num_values)
        signed = unsigned.to(torch.int32) + self._quant_min
        return signed.to(dtype) * scale

    def _gather_local_kv_cache(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_blocks = block_table >= 0
        if not valid_blocks.any():
            empty_shape = (0, kv_cache.shape[1], kv_cache.shape[2], self.headdim)
            return (
                torch.empty(empty_shape, dtype=dtype, device=kv_cache.device),
                torch.empty(
                    (0, kv_cache.shape[1], kv_cache.shape[2], self.head_size_v),
                    dtype=dtype,
                    device=kv_cache.device,
                ),
                torch.empty_like(block_table),
            )

        used_block_ids = torch.unique(block_table[valid_blocks].to(torch.int64), sorted=True)
        local_block_table = torch.full_like(block_table, -1)
        local_block_table[valid_blocks] = torch.searchsorted(
            used_block_ids,
            block_table[valid_blocks].to(torch.int64),
        ).to(block_table.dtype)

        packed_cache = kv_cache.index_select(0, used_block_ids)
        key_cache = self._dequantize_tensor(
            packed_cache[..., : self.packed_k_bytes],
            packed_cache[..., self._k_scale_start : self._k_scale_start + self.scale_bytes],
            self.headdim,
            dtype,
        ).contiguous()
        value_cache = self._dequantize_tensor(
            packed_cache[..., self._value_start : self._value_start + self.packed_v_bytes],
            packed_cache[..., self._v_scale_start : self._v_scale_start + self.scale_bytes],
            self.head_size_v,
            dtype,
        ).contiguous()
        return key_cache, value_cache, local_block_table

    def _gather_local_key_cache(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_blocks = block_table >= 0
        if not valid_blocks.any():
            empty_k = torch.empty(
                (0, kv_cache.shape[1], kv_cache.shape[2], self.headdim),
                dtype=dtype,
                device=kv_cache.device,
            )
            return (
                empty_k,
                torch.empty((0,), dtype=torch.int64, device=kv_cache.device),
                torch.empty_like(block_table),
            )
        used_block_ids = torch.unique(block_table[valid_blocks].to(torch.int64), sorted=True)
        local_block_table = torch.full_like(block_table, -1)
        local_block_table[valid_blocks] = torch.searchsorted(
            used_block_ids,
            block_table[valid_blocks].to(torch.int64),
        ).to(block_table.dtype)
        packed_cache = kv_cache.index_select(0, used_block_ids)
        key_cache = self._dequantize_tensor(
            packed_cache[..., : self.packed_k_bytes],
            packed_cache[..., self._k_scale_start : self._k_scale_start + self.scale_bytes],
            self.headdim,
            dtype,
        ).contiguous()
        return key_cache, used_block_ids, local_block_table

    def _dequantize_selected_values(
        self,
        kv_cache: torch.Tensor,
        used_block_ids: torch.Tensor,
        local_block_ids: torch.Tensor,
        token_positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        packed_blocks = kv_cache.index_select(0, used_block_ids.index_select(0, local_block_ids))
        token_offsets = token_positions.to(torch.int64)
        packed_values = packed_blocks[
            torch.arange(token_offsets.shape[0], device=kv_cache.device),
            token_offsets,
        ]
        return self._dequantize_tensor(
            packed_values[
                ..., self._value_start : self._value_start + self.packed_v_bytes
            ],
            packed_values[
                ..., self._v_scale_start : self._v_scale_start + self.scale_bytes
            ],
            self.head_size_v,
            dtype,
        ).contiguous()

    def _expand_kv_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.num_queries_per_kv == 1:
            return tensor
        return tensor.repeat_interleave(self.num_queries_per_kv, dim=-2)

    def _forward_sparse_value_decode(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        key_cache, used_block_ids, local_block_table = self._gather_local_key_cache(
            kv_cache,
            attn_metadata.block_table,
            query.dtype,
        )
        for req_idx in range(attn_metadata.query_start_loc.shape[0] - 1):
            start = int(attn_metadata.query_start_loc[req_idx])
            end = int(attn_metadata.query_start_loc[req_idx + 1])
            if end <= start:
                continue
            if end - start != 1:
                raise NotImplementedError(
                    "Sparse V decode path only supports one query token per request."
                )

            seq_len = int(attn_metadata.seq_lens[req_idx])
            local_blocks = local_block_table[req_idx]
            local_blocks = local_blocks[local_blocks >= 0].to(torch.int64)
            if local_blocks.numel() == 0 or seq_len == 0:
                output[start:end].zero_()
                continue

            key_tokens = key_cache.index_select(0, local_blocks).reshape(
                -1, self.num_kv_heads, self.headdim
            )[:seq_len]
            key_tokens = self._expand_kv_heads(key_tokens)
            q = query[start]
            attn_logits = torch.einsum("hd,thd->ht", q, key_tokens) * self.scale
            attn_probs = torch.softmax(attn_logits.float(), dim=-1).to(query.dtype)
            keep_mask = attn_probs.amax(dim=0) >= SPARSE_V_ALPHA_THRESHOLD
            if not keep_mask.any():
                keep_mask[attn_probs.argmax(dim=-1)[0]] = True

            kept_positions = torch.nonzero(keep_mask, as_tuple=False).flatten()
            selected_block_ids = torch.div(
                kept_positions,
                kv_cache.shape[1],
                rounding_mode="floor",
            )
            token_offsets = torch.remainder(kept_positions, kv_cache.shape[1])
            value_tokens = self._dequantize_selected_values(
                kv_cache,
                used_block_ids,
                local_blocks.index_select(0, selected_block_ids),
                token_offsets,
                query.dtype,
            )
            value_tokens = self._expand_kv_heads(value_tokens)
            out = torch.einsum(
                "ht,thd->hd",
                attn_probs[:, keep_mask],
                value_tokens,
            )
            output[start] = out.to(output.dtype)

        return output

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
        if self._uses_dense_layer_fallback(layer):
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )
        assert output is not None, "Output tensor must be provided."
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not supported for TurboQuant backend"
            )
        if attn_metadata is None:
            return output.fill_(0)
        if self.dcp_world_size > 1:
            raise NotImplementedError(
                "TurboQuant backend does not yet support decode context parallelism."
            )
        if (
            self.turboquant_config.sparse_v
            and not attn_metadata.use_cascade
            and attn_metadata.max_query_len == 1
        ):
            return self._forward_sparse_value_decode(
                query,
                kv_cache,
                attn_metadata,
                output,
            )

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache, local_block_table = self._gather_local_kv_cache(
            kv_cache,
            attn_metadata.block_table,
            query.dtype,
        )
        descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, self.num_kv_heads)
        q_descale = layer._q_scale.expand(descale_shape)
        k_descale = layer._k_scale.expand(descale_shape)
        v_descale = layer._v_scale.expand(descale_shape)

        if not attn_metadata.use_cascade:
            sliding_window_size = (
                list(self.sliding_window) if self.sliding_window is not None else None
            )
            flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                block_table=local_block_table,
                softcap=self.logits_soft_cap,
                scheduler_metadata=attn_metadata.scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                num_splits=attn_metadata.max_num_splits,
                s_aux=self.sinks,
            )
            return output

        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=local_block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self._uses_dense_layer_fallback(layer):
            return super().do_kv_cache_update(
                layer, key, value, kv_cache, slot_mapping
            )
        if slot_mapping.numel() == 0:
            return
        valid = slot_mapping >= 0
        if not valid.any():
            return

        key = key[: slot_mapping.shape[0]][valid]
        value = value[: slot_mapping.shape[0]][valid]
        slots = slot_mapping[valid].to(torch.int64)
        block_size = kv_cache.shape[1]
        block_ids = torch.div(slots, block_size, rounding_mode="floor")
        block_offsets = torch.remainder(slots, block_size)

        packed_k, k_scale = self._quantize_tensor(key, self.headdim)
        packed_v, v_scale = self._quantize_tensor(value, self.head_size_v)
        k_scale_bytes = self._encode_scale_bytes(k_scale)
        v_scale_bytes = self._encode_scale_bytes(v_scale)

        kv_cache[
            block_ids,
            block_offsets,
            :,
            : self.packed_k_bytes,
        ] = packed_k
        kv_cache[
            block_ids,
            block_offsets,
            :,
            self._value_start : self._value_start + self.packed_v_bytes,
        ] = packed_v
        kv_cache[
            block_ids,
            block_offsets,
            :,
            self._k_scale_start : self._k_scale_start + self.scale_bytes,
        ] = k_scale_bytes
        kv_cache[
            block_ids,
            block_offsets,
            :,
            self._v_scale_start : self._v_scale_start + self.scale_bytes,
        ] = v_scale_bytes
