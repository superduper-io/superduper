from superduperdb.utils import CallableWithSecret


class Agent(CallableWithSecret):
    def __init__(self, object_, secrets, method='__call__'):
        super().__init__(secrets)
        self._method = method
        self.object = object_

    def __call__(self, *args, **kwargs):
        f = getattr(self.object, self._method)
        return f(*args, **kwargs)
