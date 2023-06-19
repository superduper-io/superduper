from sklearn.pipeline import Pipeline as BasePipeline
from superduperdb.core.model import Model


def postprocess(f):
    f.superduper_postprocess = True
    return f


class Pipeline(Model):
    def __init__(
        self,
        identifier,
        steps,
        memory=None,
        verbose=False,
        encoder=None,
    ):
        standard_steps = [
            i
            for i, s in enumerate(steps)
            if not getattr(s[1], 'superduper_postprocess', False)
        ]
        postprocess_steps = [
            i
            for i, s in enumerate(steps)
            if getattr(s[1], 'superduper_postprocess', False)
        ]

        if postprocess_steps:
            msg = 'Postprocess steps must go after preprocess steps'
            assert max(standard_steps) < min(postprocess_steps), msg

        pipeline = BasePipeline(
            [steps[i] for i in standard_steps], memory=memory, verbose=verbose
        )
        self.postprocess_steps = [steps[i] for i in postprocess_steps]
        Model.__init__(self, pipeline, identifier, encoder=encoder)

    def score(self, X, y=None, **predict_params):
        return self.object.score(X, y=y, **predict_params)

    def fit(self, X, y):
        return self.object.fit(X, y)

    def predict(self, X, **predict_params):
        out = self.object.predict(X, **predict_params).tolist()
        if self.postprocess_steps:
            for s in self.postprocess_steps:
                if hasattr(s[1], 'transform'):
                    out = s[1].transform(s)
                elif callable(s[1]):
                    out = s[1](out)
                else:
                    raise Exception('Unexpected postprocessing transform')
        return out
