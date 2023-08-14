try:
    import torch

    from superduperdb.ext.torch.model import TorchModel
    from superduperdb.ext.torch.tensor import tensor

except ImportError:
    torch = tensor = TorchModel = None

import pytest

skip_torch = pytest.mark.skipif(torch is None, reason='Torch not installed')
