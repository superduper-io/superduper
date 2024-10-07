import importlib
import inspect
import typing as t

from superduper.components.schema import Schema


def _decorator(f):
    from superduper import ObjectModel

    def dec(
        *args,
        identifier=None,
        datatype=None,
        model_update_kwargs: t.Optional[t.Dict] = None,
        output_schema: t.Optional[Schema] = None,
        **kwargs,
    ):
        model_update_kwargs = model_update_kwargs or {}
        return ObjectModel(
            identifier=identifier or f.__name__,
            object=f(*args, **kwargs),
            datatype=datatype,
            model_update_kwargs=model_update_kwargs,
            output_schema=output_schema,
        )

    return dec


class _Package:
    def __init__(self, package_name: t.Optional[str] = None, package=None):
        if package is None:
            assert package_name is not None
            self.package = importlib.import_module(package_name)
        else:
            self.package = package

    def __getattr__(self, item):
        object_ = getattr(self.package, item)
        if inspect.ismodule(object_):
            return _Package(package=object_)
        return _decorator(object_)


def __getattr__(name):
    try:
        return _Package(name)
    except ImportError:
        raise AttributeError(f"module {__file__} has no attribute {name}")
