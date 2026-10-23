from asr1bit.replace import COMPONENTS, POLICIES, component_of

NAMES = {
    "q": "model.layers.0.self_attn.q_proj",
    "k": "model.layers.0.self_attn.k_proj",
    "v": "model.layers.0.self_attn.v_proj",
    "o": "model.layers.0.self_attn.o_proj",
    "ffn_in": "model.layers.0.mlp.gate_proj",
    "ffn_out": "model.layers.0.mlp.down_proj",
}


def test_component_lookup_is_decoder_only():
    for key, name in NAMES.items():
        assert component_of(name) == key
    assert component_of("audio_tower.layers.0.self_attn.q_proj") is None
    assert component_of("model.layers.0.input_layernorm.weight") is None


def test_every_component_has_a_policy():
    for key in COMPONENTS:
        policy = POLICIES[f"decoder_{key}"]
        assert policy(NAMES["q" if key == "attn" else "ffn_out" if key == "ffn" else key])
        assert not policy("audio_tower.layers.0.self_attn.q_proj")


def test_attn_and_ffn_policies_are_disjoint():
    attn, ffn = POLICIES["decoder_attn"], POLICIES["decoder_ffn"]
    for name in NAMES.values():
        assert attn(name) != ffn(name)
