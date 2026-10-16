"""Stage-5 export stubs: the locked interface exists before the kernels do.

`deploy/export.py` is a Stage-5 skeleton: the three entry points exist but
raise, by design. This test pins their *shape* so that the eventual
implementation inherits the documented contract instead of being free to
invent one (see docs/PLANNING.md §8 and rules.md "locked interfaces").
"""

import pytest


def test_locked_interface_is_present():
    from asr1bit.deploy import export

    for name in ("pack_ternary", "export_gguf"):
        fn = getattr(export, name, None)
        assert callable(fn), f"missing locked export entry point: {name}"


@pytest.mark.parametrize("name", ["pack_ternary", "export_gguf"])
def test_stage5_stubs_raise_until_implemented(name):
    from asr1bit.deploy import export

    fn = getattr(export, name)
    with pytest.raises(NotImplementedError):
        # export_gguf takes (model, out_path); the others take one payload arg
        fn(*(None, None) if name == "export_gguf" else (None,))
