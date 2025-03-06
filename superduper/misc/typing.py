import typing as t

File = t.NewType('File', t.AnyStr)
Dill = t.NewType('Dill', t.Callable)
Pickle = t.NewType('Pickle', t.Callable)
PickleEncoder = t.NewType('PickleEncoder', t.Any)
SDict = t.NewType('SDict', t.Dict)
SList = t.NewType('SList', t.List)
FDict = t.NewType('FDict', t.Dict)
JSON = t.NewType('JSON', t.Union[t.Dict, t.List, str, int])
LeafType = t.NewType('LeafType', t.Any)
