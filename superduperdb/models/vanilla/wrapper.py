import multiprocessing
from superduperdb.core.model import Model


class Function(Model):
    vanilla = True

    def predict_one(self, x, **kwargs):
        return self.object.artifact(x, **kwargs)

    def _predict(self, docs, num_workers=0, **kwargs):
        outputs = []
        if not isinstance(docs, list):
            return self.predict_one(docs)
        if num_workers:
            pool = multiprocessing.Pool(processes=num_workers)
            for r in pool.map(self.object.artifact, docs):
                outputs.append(r)
            pool.close()
            pool.join()
        else:
            for r in docs:
                outputs.append(self.object.artifact(r))
        return outputs
