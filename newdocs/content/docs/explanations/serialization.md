# Serialization

SuperDuperDB makes extensive use of various serialization techniques in Python, 
in order to save the following objects to the `DB`, which inherit
from `superduperdb.container.base.Component`

- [Models](models)
- [Vector Indexes](vectorsearch)
- [Encoders](encoders)

Since these are compound objects consisting of multiple component parts, we utilize a hybrid 
serialization scheme, in order to save these objects. The scheme is based on:

- Classes wrapped with `@dataclasses.dataclass`
- Items in `object.dict()` which are not serializable as JSON, serialized using:
  - `dill`
  - `pickle`
  - `torch.save`

Important items which can't be saved using JSON, are signified inside the object
with the wrapper `superduperdb.core.artifact.Artifact`. This wrapper carries
around the serialization method necessary to save it's wrapped object in the 
[artifact store](artifactstore).

At save time the following algorithm is executed:

1. The user passes `my_object` a descendant of `Component` to be saved in `db.add`
2. SuperDuperDB executes `d = my_object.to_dict()`.
3. SuperDuperDB extracts the `Artifact` instances out of `d` and saves these
   in the [artifact store](artifactstore).
4. SuperDuperDB saves the `d` which includes references to the `Artifact` instances
   to the [metadata store](metadata).
