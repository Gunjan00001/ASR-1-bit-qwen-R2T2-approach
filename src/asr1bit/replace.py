"""Layer policy: choose which linear layers become BitLinear (Stage 1, T1.3).

Default policy binarizes the decoder attention projections only
(``q/k/v/o``). Encoder, embeddings, LM head, and norms stay high precision.
Policies match on qualified module names; the audio encoder is excluded by
prefix so its ``self_attn`` (if named the same) is left alone.

Locked interface: ``apply_layer_policy(model, policy) -> model``.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from torch import nn

from asr1bit.bitlinear import BitLinear

_ATTN = re.compile(r"(?:^|\.)self_attn\.(?:q|k|v|o)_proj$")
_FFN = re.compile(r"(?:^|\.)mlp\.(?:gate|up|down)_proj$")
_ENCODER_TOKENS = ("audio_tower", "audio_encoder", ".encoder.")


def _is_encoder(name: str) -> bool:
    return any(token in name for token in _ENCODER_TOKENS)


def _decoder_attn(name: str) -> bool:
    return bool(_ATTN.search(name)) and not _is_encoder(name)


def _decoder_attn_ffn(name: str) -> bool:
    return bool(_ATTN.search(name) or _FFN.search(name)) and not _is_encoder(name)


POLICIES: dict[str, Callable[[str], bool]] = {
    "none": lambda name: False,
    "decoder_attn": _decoder_attn,
    "decoder_attn_ffn": _decoder_attn_ffn,
}


def _resolve_policy(policy: str | Callable[[str], bool]) -> Callable[[str], bool]:
    if callable(policy):
        return policy
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; expected one of {sorted(POLICIES)}")
    return POLICIES[policy]


def _parent_and_attr(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, attr = name.rpartition(".")
    parent = model.get_submodule(parent_name)
    return parent, attr


def apply_layer_policy(
    model: nn.Module,
    policy: str | Callable[[str], bool] = "decoder_attn",
    *,
    bit_width: float = 1,
    quantize_activations: bool = True,
    group_size: int | None = None,
) -> nn.Module:
    """Replace selected ``nn.Linear`` layers with :class:`BitLinear` in place."""
    predicate = _resolve_policy(policy)
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not isinstance(module, BitLinear) and predicate(name):
            parent, attr = _parent_and_attr(model, name)
            setattr(
                parent,
                attr,
                BitLinear.from_linear(
                    module,
                    bit_width=bit_width,
                    quantize_activations=quantize_activations,
                    group_size=group_size,
                ),
            )
            replaced += 1
    model.bitlinear_replaced = replaced
    return model


def count_bitlinear(model: nn.Module) -> int:
    """Number of :class:`BitLinear` modules in ``model``."""
    return sum(1 for module in model.modules() if isinstance(module, BitLinear))
