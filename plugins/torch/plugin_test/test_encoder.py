from superduper.base.schema import Schema

from superduper_torch.encoder import Tensor


def test_build_tensor_datatype():

    s = Schema.build(tensor='superduper_torch.Tensor[float64:2x3]')

    assert isinstance(s.fields['tensor'], Tensor)
