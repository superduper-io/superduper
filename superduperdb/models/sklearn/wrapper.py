from sklearn.pipeline import Pipeline as BasePipeline
from superduperdb.models.base import SuperDuperModel


class Pipeline(BasePipeline, SuperDuperModel):
    def __init__(self, steps, memory=None,
                 verbose=False, postprocessor=None):

        BasePipeline.__init__(self, steps=steps, memory=memory, verbose=verbose)
        SuperDuperModel.__init__(self)
        self.postprocessor = postprocessor

    def predict_one(self, X, **predict_params):
        return self.predict([X], **predict_params)[0]

    def predict(self, X, **predict_params):
        out = super().predict(X, **predict_params).tolist()
        if self.postprocessor is not None:
            return [self.postprocessor(x) for x in out]
        return list(out)