from superduperdb import cf
from superduperdb.utils import ArgumentDefaultDict


class BaseDatabase:
    def __init__(self):

        self.models = ArgumentDefaultDict(lambda x: self._load_model(x))
        self.functions = ArgumentDefaultDict(lambda x: self._load_object('function', x))
        self.preprocessors = ArgumentDefaultDict(lambda x: self._load_object('preprocessor', x))
        self.postprocessors = ArgumentDefaultDict(lambda x: self._load_object('postprocessor', x))
        self.types = ArgumentDefaultDict(lambda x: self._load_object('type', x))
        self.splitters = ArgumentDefaultDict(lambda x: self._load_object('splitter', x))
        self.objectives = ArgumentDefaultDict(lambda x: self._load_object('objective', x))
        self.measures = ArgumentDefaultDict(lambda x: self._load_object('measure', x))
        self.metrics = ArgumentDefaultDict(lambda x: self._load_object('metric', x))

        self.remote = cf.get('remote', False)