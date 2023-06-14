import dataclasses as dc
import inspect
import superduperdb as s
import typing as t


class NotJSONableError(ValueError):
    pass


@dc.dataclass(frozen=True)
class RegistryEntry:
    name: str
    parameter_types: t.Sequence[t.Type]
    result_type: t.Type


@dc.dataclass
class Registry:
    entries: t.Dict[str, RegistryEntry] = dc.field(default_factory=dict)
    Parameter: t.Optional[t.Type] = None
    Result: t.Optional[t.Type] = None

    def register(self, method: t.Callable) -> None:
        """
        Registers a method and its signature.
        """
        name = method.__name__
        assert name not in self.entries, f'name={name} already registered'

        sig = inspect.signature(method)
        params = list(sig.parameters.items())
        if not inspect.ismethod(method):
            if not (params and params[0][0] == 'self'):
                raise NotJSONableError(f'{method} not an instance method')
            params = params[1:]

        parameter_types = tuple(value.annotation for _, value in params)
        result_type = sig.return_annotation

        def is_model(t, parent=s.JSONable):
            return t is parent or any(is_model(c, parent) for c in t.__bases__)

        types = set(parameter_types + (result_type,))
        if non_models := [t for t in types if not is_model(t)]:
            raise NotJSONableError(f'Not serializable: {non_models}')

        def unite(a, b):
            return b if a is None else t.Union[a, b]

        for r in parameter_types:
            self.Parameter = unite(self.Parameter, r)
        self.Result = unite(self.Result, result_type)
        self.entries[name] = RegistryEntry(name, parameter_types, result_type)

    def execute(
        self, obj: t.Any, method: str, args: t.Sequence[s.JSONable]
    ) -> s.JSONable:
        """Executes a query with zero or more parameters"""

        method_impl = getattr(obj, method, None)
        assert method_impl is not None, f'No database method named {method}'

        entry = self.entries.get(method)
        assert entry, f'No method named {method}'

        # Going through `execute` destroys most of the type information, so we need to
        # validate "by hand" here
        types = entry.parameter_types

        if not types and len(args) == 1 and args[0] is None:
            # Special case, needed for endpoint /execute-one
            args = ()
        elif len(args) != len(types):
            assert False, f'In {method}, expected {len(args)} objects, got {len(types)}'

        arg_types = enumerate(zip(args, types))
        if bad := [(i, a, t) for i, (a, t) in arg_types if not isinstance(a, t)]:
            it = (f'In param {i} of {method}, expected {t}, got {a}' for i, a, t in bad)
            assert False, '\n'.join(it)

        return method_impl(*args)
