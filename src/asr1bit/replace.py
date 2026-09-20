"""Layer policy: choose which linear layers become BitLinear.

Stage 1, T1.3.

Binarize: decoder q/k/v/o + gate/up/down projections.
Keep higher precision: embeddings, LM head, LayerNorm/softmax, AuT encoder.

Locked interface:
    apply_layer_policy(model, policy) -> model
"""


def apply_layer_policy(model, policy):
    """Replace selected linear layers with BitLinear according to ``policy``."""
    raise NotImplementedError("Stage 1, T1.3")
