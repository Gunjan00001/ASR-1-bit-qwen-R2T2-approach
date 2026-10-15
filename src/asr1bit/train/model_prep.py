"""Small model-preparation helpers used before training or evaluation."""
from __future__ import annotations


def patch_outer_forward(model) -> None:
    """Route the outer forward to ``model.thinker`` (official Qwen3-ASR API)."""
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker"):
        raise RuntimeError("model has no .thinker; cannot patch forward")

    def forward(self, input_ids=None, attention_mask=None, input_features=None,
                feature_attention_mask=None, labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features, feature_attention_mask=feature_attention_mask,
            labels=labels, **kwargs)

    cls.forward = forward
    cls._forward_patched = True
