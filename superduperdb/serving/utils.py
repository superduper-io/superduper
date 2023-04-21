from superduperdb import cf
from superduperdb.types.utils import convert_from_bytes_to_types


def maybe_login_required(auth, service):
    """
    Require login depending on the contents of the config file.:w

    :param auth: basic auth instance
    :param service: name of the service on question
    """
    def decorator(f):
        if 'user' in cf[service]:
            return auth.login_required(f)
        return f
    return decorator


def encode_args_kwargs(database, args, kwargs, positional_convertible, keyword_convertible):
    args = list(args)
    for i, arg in enumerate(args):
        if positional_convertible[i]:
            args[i] = database.convert_from_types_to_bytes(arg)
    for k in kwargs:
        if keyword_convertible[k]:
            kwargs[k] = database.convert_from_types_to_bytes(kwargs[k])
    return tuple(args), kwargs


def decode_args_kwargs(database, args, kwargs, positional_convertible, keyword_convertible):
    args = list(args)
    for i, arg in enumerate(args):
        if positional_convertible[i]:
            args[i] = convert_from_bytes_to_types(arg, database.types)
    for k in kwargs:
        if keyword_convertible[k]:
            kwargs[k] = convert_from_bytes_to_types(kwargs[k], database.types)
    return tuple(args), kwargs


def encode_ids_parameters(args, kwargs, positional_convertible, keyword_convertible):
    args = list(args)[:]
    for i, arg in enumerate(args):
        if positional_convertible[i]:
            args[i] = [str(_id) for _id in arg]

    for k, v in kwargs.items():
        if keyword_convertible[k]:
            kwargs[k] = [str(_id) for _id in v]

    return args, kwargs