import typing as t

from superduper import Component as C
from superduper.components.datatype import *

File = t.NewType('File', t.AnyStr)
Blob = t.NewType('Blob', t.Callable)
Artifact = t.NewType('Artifact', t.Callable)
Dill = t.NewType('Dill', t.Callable)
Pickle = t.NewType('Pickle', t.Callable)
SDict = t.NewType('SDict', t.Dict)
SList = t.NewType('SList', t.List)
Component = t.NewType('Component', C)
