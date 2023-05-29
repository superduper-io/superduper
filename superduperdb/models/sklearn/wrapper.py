from sklearn.pipeline import Pipeline as BasePipeline
from superduperdb.core.model import Model


class Pipeline(BasePipeline, Model):
    def __init__(self, steps, identifier, memory=None, verbose=False, postprocessor=None, type=None):
        BasePipeline.__init__(self, steps=steps, memory=memory, verbose=verbose)
        Model.__init__(self, None, identifier, type=type)
        self.postprocessor = postprocessor

    def predict_one(self, X, **predict_params):
        return self.predict([X], **predict_params)[0]

    def predict(self, X, **predict_params):
        out = super().predict(X, **predict_params).tolist()
        if self.postprocessor is not None:
            return [self.postprocessor(x) for x in out]
        return list(out)
