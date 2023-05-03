import torch
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline as BasePipeline
from transformers import Pipeline as HuggingPipeline
from sentence_transformers import SentenceTransformer


class SuperDuperModel:
    def predict_one(self, r, **kwargs):
        raise NotImplementedError

    def predict(self, docs, **kwargs):
        raise NotImplementedError


def wrap_model(model, preprocess=None, postprocess=None):
    if isinstance(model, SuperDuperModel):
        return model
    elif isinstance(model, HuggingPipeline):
        from .transformers.wrapper import TransformersWrapper
        return TransformersWrapper(model)
    elif isinstance(model, SentenceTransformer):
        from .sentence_transformers.wrapper import SentenceTransformerWrapper
        return SentenceTransformerWrapper(model)
    elif isinstance(model, torch.nn.Module):
        from .torch.wrapper import SuperDuperWrapper
        return SuperDuperWrapper(model, preprocess=preprocess, postprocess=postprocess)
    elif isinstance(model, BaseEstimator):
        from .sklearn.wrapper import Pipeline
        if preprocess is not None:
            return Pipeline(steps=[
                ('preprocess', preprocess),
                ('estimator', model)
            ], postprocessor=postprocess)
        else:
            return Pipeline(steps=[('estimator', model)], postprocessor=postprocess)
    elif isinstance(model, BasePipeline):
        from .sklearn.wrapper import Pipeline
        return Pipeline(steps=model.steps, memory=model.memory, verbose=model.verbose)
    else:
        assert callable(model)
        from .vanilla.wrapper import FunctionWrapper
        return FunctionWrapper(model)
