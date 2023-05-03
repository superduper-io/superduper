import multiprocessing
from superduperdb.models.base import SuperDuperModel


class FunctionWrapper(SuperDuperModel):
    def __init__(self, f):
        self.f = f

    def predict_one(self, x, **kwargs):
        return self.f(x, **kwargs)

    def predict(self, docs, num_workers=0):
        outputs = []
        if num_workers:
            pool = multiprocessing.Pool(processes=num_workers)
            for r in pool.map(self.f, docs):
                outputs.append(r)
            pool.close()
            pool.join()
        else:
            for r in docs:  # pragma: no cover
                outputs.append(self.f(r))
        return outputs
