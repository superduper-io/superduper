# Class hierarchy of user-facing classes

## Datalayer

The `Datalayer` class, an instance of which we refer to throughout this 
documentation as `db`, is the key entrypoint via which developers
may connect to their data-infrastructure and additional connect
AI functionality to their data-infrastructure:

### Connect with `superduper`

The `Datalayer` connects to data, with the [`superduper` function](../core_api/connect).

***`.apply`***

AI components may be applied to the built `Datalayer` [with `.apply`](../core_api/apply).

***`.execute`***

The data and AI outputs are accessible with queries and AI models 
using the `.execute` method. This can include standard database queries,
vector-search queries (which include model inference) and pure model computations.
See [here](../core_api/execute).

***`.show`***

`.show` provides information about components applied with `.apply`.

***`.load`***

`.load` reloads components applied with `.apply`.

## `Document`

`Document` is a wrapper around standard Python `dict` instances, 
but which can encode their contained fields as a mixture of JSON
and pure `bytes`. This mechanism can in principle handle any information 
which Python can handle.

Since most databases can handle this type of information, this makes
`Document` a crucial piece in connecting AI (which operates over a range of information)
and the database.

***`.encode`***

Convert the `Document` instance to a dictionary of JSON-able information 
and `bytes`.

***`.decode`***

Convert a dictionary of JSON-able information 
and `bytes` to a `Document` contained the full range of Python objects.

## `Leaf`

A `Leaf` is the wrapper around a Python object, which carries with 
it the mechanism via which it converts the object to a mixture 
of JSON and `bytes`.

Here are the key `Leaf` types:

### `Serializable`

An extension of Python `dataclasses`, but easier to get the original class back 
from the serialized dictionary form.

### `Component`

Builds on top of `Serializable` but also allows the additional presence of non-JSON-able content via `_BaseEncodable`:

| Class | Description |
| --- | --- |
| `Encodable` |  |
| `Artifact` |  |
| `LazyArtifact` |  |
| `File` |  |


### `_BaseEncodable`

| Class | Description |
| --- | --- |
| `Encodable` |   |
| `Artifact` |   |
| `LazyArtifact` |   |
| `File` |   |

### Code

...
