import typing as t

File = t.NewType('File', str)
Blob = t.NewType('Blob', t.Callable)
Dill = t.NewType('Dill', t.Callable)
Pickle = t.NewType('Pickle', t.Callable)
JSON = t.NewType('JSON', t.Dict)
