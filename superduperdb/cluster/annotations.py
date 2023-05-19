import inspect
from typing import Tuple as BaseTuple

from bson import ObjectId


def encode_args(database, signature, args):
    parameters = signature.parameters
    positional_parameters = [
        k
        for k in parameters
        if parameters[k].default == inspect.Parameter.empty
        and parameters[k].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        and k != 'self'
    ]
    out = list(args)[:]
    for i, arg in enumerate(args):
        if isinstance(parameters[positional_parameters[i]].annotation, Convertible):
            out[i] = parameters[positional_parameters[i]].annotation.encode(
                database, arg
            )
    return out


def encode_kwargs(database, signature, kwargs):
    parameters = signature.parameters
    keyword_parameters = [
        param
        for param, details in parameters.items()
        if details.default != inspect.Parameter.empty
    ]
    out = kwargs.copy()
    for k in keyword_parameters:
        if isinstance(parameters[k].annotation, Convertible):
            out[k] = parameters[k].annotation.encode(database, kwargs[k])
    return out


def encode_result(database, signature, result):
    if isinstance(signature.return_annotation, Convertible):
        return signature.return_annotation.encode(database, result)
    return result


def decode_args(database, signature, args):
    parameters = signature.parameters
    positional_parameters = [
        k
        for k in parameters
        if parameters[k].default == inspect.Parameter.empty
        and parameters[k].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        and k != 'self'
    ]
    out = list(args)[:]
    for i, arg in enumerate(args):
        if isinstance(parameters[positional_parameters[i]].annotation, Convertible):
            out[i] = parameters[positional_parameters[i]].annotation.decode(
                database, arg
            )
    return out


def decode_kwargs(database, signature, kwargs):
    parameters = signature.parameters
    keyword_parameters = [
        param
        for param, details in parameters.items()
        if details.default != inspect.Parameter.empty
    ]
    out = kwargs.copy()
    for k in keyword_parameters:
        if isinstance(parameters[k].annotation, Convertible):
            out[k] = parameters[k].annotation.decode(database, kwargs[k])
    return out


def decode_result(database, signature, result):
    if isinstance(signature.return_annotation, Convertible):
        return signature.return_annotation.decode(database, result)
    return result


class Convertible:
    def _encode(self, database, x):
        return database.convert_from_types_to_bytes(x)

    def _decode(self, database, x):
        return database.convert_from_bytes_to_types(x)

    def encode(self, database, x):
        if x is None:
            return
        return self._encode(database, x)

    def decode(self, database, x):
        if x is None:
            return
        return self._decode(database, x)


class ObjectIdConvertible(Convertible):
    def _encode(self, database, x):
        return str(x)

    def _decode(self, database, x):
        return ObjectId(x)


class List(Convertible):
    def __init__(self, item_type):
        self.item_type = item_type

    def _encode(self, database, x):
        out = []
        for y in x:
            out.append(self.item_type.encode(database, y))
        return out

    def _decode(self, database, x):
        out = []
        for y in x:
            out.append(self.item_type.decode(database, y))
        return out


class Tuple(BaseTuple, Convertible):
    """
    >>> from typing import Any
    >>> database = lambda: None
    >>> database.convert_from_types_to_bytes = lambda x: x + ': encoded'
    >>> Tuple([Convertible(), Any]).encode(database, ('a test', 'another'))
    ('a test: encoded', 'another')

    """

    def __init__(self, item_types):
        self.item_types = item_types

    def _encode(self, database, x):
        out = []
        for t, y in zip(self.item_types, x):
            if hasattr(t, 'encode'):
                try:
                    out.append(t.encode(database, y))
                except Exception as e:
                    # import pdb; pdb.set_trace()
                    raise e
            else:
                out.append(y)
        return tuple(out)

    def _decode(self, database, x):
        out = []
        for t, y in zip(self.item_types, x):
            if hasattr(t, 'decode'):
                out.append(t.decode(database, y))
            else:
                out.append(y)
        return tuple(out)
