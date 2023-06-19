from sklearn.pipeline import Pipeline as BasePipeline
from superduperdb.core.model import Model


class Pipeline(Model):
    def __init__(
        self,
        steps,
        identifier,
        memory=None,
        verbose=False,
        postprocessor=None,
        encoder=None,
    ):
        pipeline = BasePipeline(steps, memory=memory, verbose=verbose)
        Model.__init__(self, pipeline, identifier, encoder=encoder)
        self.postprocessor = postprocessor

    def score(self, X, y=None, **predict_params):
        return self.object.score(X, y=y, **predict_params)

    def fit(self, X, y):
        return self.object.fit(X, y)

    def predict_one(self, X, **predict_params):
        return self.object.predict([X], **predict_params)[0]

    def predict(self, X, **predict_params):
        out = self.object.predict(X, **predict_params).tolist()
        if self.postprocessor is not None:
            return [self.postprocessor(x) for x in out]
        return list(out)
