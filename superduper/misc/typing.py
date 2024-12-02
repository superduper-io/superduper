import typing as t

from superduper.components.datatype import *

File = t.NewType('File', t.AnyStr)
Blob = t.NewType('Blob', t.Callable)
Dill = t.NewType('Dill', t.Callable)
Pickle = t.NewType('Pickle', t.Callable)
JSON = t.NewType('JSON', t.Dict)
