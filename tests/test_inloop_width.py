from torch import nn

from asr1bit.bitlinear import BitLinear
from asr1bit.train.inloop_eval import set_precision


def test_set_precision_accepts_two_bit():
    model = nn.Sequential(BitLinear(8, 4, bit_width=1.58))
    set_precision(model, 2.0, 1.0)
    assert model[0].bit_width == 2.0
