import multiprocessing
import typing as t

from superduperdb.core.model import Model


class FunctionWrapper(Model):
    def predict_one(self, x: t.Any, **kwargs: t.Any) -> t.Any:
        return self.object(x, **kwargs)

    def predict(self, docs: t.List[t.Dict], num_workers: int = 0) -> t.List:
        outputs = []
        if num_workers:
            pool = multiprocessing.Pool(processes=num_workers)
            for r in pool.map(self.object, docs):
                outputs.append(r)
            pool.close()
            pool.join()
        else:
            for r in docs:
                outputs.append(self.object(r))
        return outputs
