import torch


class Container(torch.nn.Module):
    """
    Class wrapping a ``torch.nn.Module`` adding preprocessing and postprocessing

    :param preprocessor: preprocessing function
    :param forward: forward pass
    :param postprocessor: postprocessing function
    """
    def __init__(self, preprocessor=None, forward=None, postprocessor=None):
        super().__init__()
        self._preprocess = preprocessor
        self._forward = forward
        self._postprocess = postprocessor

    def preprocess(self, *args, **kwargs):
        if self._preprocess is not None:
            return self._preprocess(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    def postprocess(self, *args, **kwargs):
        if self._postprocess is not None:
            return self._postprocess(*args, **kwargs)


class TrivialContainer:
    def __init__(self, preprocess=None):
        self.preprocess = preprocess


def create_container(preprocessor=None, forward=None, postprocessor=None):
    if forward is not None:
        assert isinstance(forward, torch.nn.Module)
    if postprocessor is not None:
        assert forward is not None
    if forward is None:
        return TrivialContainer(preprocessor)

    if preprocessor is None:
        preprocessor = lambda x: x
    if postprocessor is None:
        postprocessor = lambda x: x
    return Container(preprocessor=preprocessor, forward=forward, postprocessor=postprocessor)


