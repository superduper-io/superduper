import importlib
import json
import os
import shutil


class Jsonable:
    ext = ''

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def data(self):
        raise NotImplementedError

    @data.setter
    def data(self, value):
        raise NotImplementedError

    def _list_to_serialized(self, l_):
        out = []
        v = {}
        g = {}
        for d in l_:
            if isinstance(d, list):
                out.append(self._list_to_serialized(d))
            elif isinstance(d, Jsonable):
                defn, sg, tmp = d.to_dict(root=False)
                out.append('$' + str(id(d)))
                g.update(sg)
                v.update(tmp)
            else:
                out.append(d)
        return out, g, v

    @staticmethod
    def _parse_list(l_, g, v):
        out = []
        for d in l_:
            if isinstance(d, str) and d.startswith('$'):
                out.append(Jsonable.from_dict(d[1:], g, v))
            elif isinstance(d, list):
                out.append(Jsonable._parse_list(d, g, v))
            else:
                out.append(d)
        return out

    def to_dict(self, root=False):
        values = {}
        graph = {}
        kwargs = self.kwargs.copy()
        id_ = str(id(self)) if not root else 'main'
        for k, d in kwargs.items():
            if isinstance(d, Jsonable):
                _, subgraph, subvalues = d.to_dict(root=False)
                kwargs[k] = '$' + str(id(d))
                graph.update(subgraph)
                values.update(subvalues)
            elif isinstance(d, list):
                subkwargs, subgraph, subvalues = self._list_to_serialized(d)
                kwargs[k] = subkwargs
                graph.update(subgraph)
                values.update(subvalues)
        out = {
            'module': self.__module__,
            'cls': self.__class__.__name__,
            'kwargs': kwargs,
        }
        graph[id_] = out
        if out['module'] == '__main__':
            raise Exception('Can\'t save a class in "__main__"')
        try:
            values[id_] = {'data': self.data, 'ext': self.ext}
        except NotImplementedError:
            pass
        return out, graph, values

    def save(self, path, force=False):
        if os.path.exists(path) and not force:
            raise Exception(f'File/ directory {path} already exists! Set force=True '
                            'if you\'d like to go ahead anyway.')

        if force:
            shutil.rmtree(path)
        os.makedirs(path)
        d, g, v = self.to_dict(root=True)
        with open(path + '/ai.json', 'w') as f:
            json.dump(g, f, indent=2)
        for id_ in v:
            with open(f'{path}/{id_}{v[id_]["ext"]}', 'wb') as f:
                f.write(v[id_]['data'])

    @staticmethod
    def load(path):
        with open(path + '/ai.json') as f:
            g = json.load(f)
        v = {}
        for file in os.listdir(path):
            id_ = file.split('.')[0]
            if id_ == 'ai':
                continue
            with open(path + '/' + file, 'rb') as f:
                data = f.read()
            v[id_] = {'data': data}
        return Jsonable.from_dict('main', g, v)

    @staticmethod
    def from_dict(id_, graph, values):
        defn = graph[id_]
        if not isinstance(defn, dict):
            return defn
        module = importlib.import_module(defn['module'])
        cls = getattr(module, defn['cls'])
        kwargs = defn['kwargs'].copy()
        for k, val in kwargs.items():
            if isinstance(val, dict) and {'cls', 'kwargs', 'id'}.issubset(set(val.keys())):
                kwargs[k] = Jsonable.from_dict(val[1:], graph, values)
            elif isinstance(val, list):
                kwargs[k] = Jsonable._parse_list(val, graph, values)
        instance = cls(**kwargs)
        if id_ in values:
            instance.data = values[id_]['data']
        graph[id_] = instance
        return instance
