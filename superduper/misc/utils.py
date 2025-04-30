import hashlib
import typing as t


def str_shape(shape: t.Sequence[int] | int) -> str:
    """Convert a shape to a string.

    :param shape: The shape to convert.
    """
    if isinstance(shape, int):
        return str(shape)
    if not shape:
        raise ValueError('Shape was empty')
    return 'x'.join(str(x) for x in shape)


def merge_dicts(r: t.Dict, s: t.Dict) -> dict:
    """Merge two dictionaries recursively.

    :param r: The first dictionary.
    :param s: The second dictionary.

    >>> r = {'foo': {'bar': 1, 'baz': 2}, 'qux': 3}
    >>> s = {'foo': {'bar': 4, 'quux': 5}, 'quux': 6}
    >>> merge_dicts(r, s)
    {'foo': {'bar': 4, 'baz': 2, 'quux': 5}, 'qux': 3, 'quux': 6}
    """
    for k, v in s.items():
        if isinstance(v, dict) and k in r:
            r[k] = merge_dicts(r[k], v)
        else:
            r[k] = v
    return r


# TODO move to plugins
def format_prompt(X: str, prompt: str, context: t.Optional[t.List[str]] = None) -> str:
    """Format a prompt with the given input and context.

    :param X: The input to format the prompt with.
    :param prompt: The prompt to format.
    :param context: The context to format the prompt with.
    """
    format_params = {}
    if '{input}' in prompt:
        format_params['input'] = X
    else:
        prompt += X

    if '{context}' in prompt:
        if isinstance(context, (list, tuple)):
            context = '\n'.join(context)

        if context:
            format_params['context'] = context
        else:
            raise ValueError(f'A context is required for prompt {prompt}')

    return prompt.format(**format_params)


def hash_item(item: t.Any) -> str:
    """Hash an item.

    :param item: The item to hash.
    """
    if item is None:
        return hashlib.sha256(('<NoneType>' + str(item)).encode()).hexdigest()
    if isinstance(item, bytearray):
        return hashlib.sha256(item).hexdigest()
    if isinstance(item, str):
        return hashlib.sha256(str(item).encode()).hexdigest()
    if isinstance(item, float):
        return hashlib.sha256(('<float>' + str(item)).encode()).hexdigest()
    if isinstance(item, int):
        return hashlib.sha256(('<int>' + str(item)).encode()).hexdigest()
    if isinstance(item, bool):
        return hashlib.sha256(('<bool>' + str(item)).encode()).hexdigest()
    if isinstance(item, (list, tuple)):
        hashes = []
        for i in item:
            hashes.append(hash_item(i))
        hashes = ''.join(hashes)
        return hashlib.sha256(hashes.encode()).hexdigest()
    if isinstance(item, dict):
        keys = sorted(item.keys())
        hashes = []
        for k in keys:
            hashes.append((hash_item(k), hash_item(item[k])))  # type: ignore[arg-type]
        return hashlib.sha256(str(hashes).encode()).hexdigest()
    return hashlib.sha256(str(item).encode()).hexdigest()


def _history_listing_to_dict(raw: str) -> dict[str, str]:
    import re

    pattern = re.compile(r"^\s*(\d+/\d+|\d+):\s*(.*)$")
    lines = raw.splitlines()
    out = {}
    i = 0
    while i < len(lines):
        m = pattern.match(lines[i])
        if m:  # found a header
            key, first = m.groups()
            if first:  # code on same line
                out[key] = first
                i += 1
            else:  # code starts next line(s)
                i += 1
                buf = []
                while i < len(lines) and not pattern.match(lines[i]):
                    buf.append(lines[i])
                    i += 1
                out[key] = "\n".join(buf)
        else:
            i += 1
    return out


def grab_source_code_ipython(cls_or_fn: t.Union[t.Type, t.Callable]) -> str:
    """Grab the source code of a class or function.

    :param cls_or_fn: The class or function
    """
    from contextlib import redirect_stdout
    from io import StringIO

    from IPython import get_ipython

    ip = get_ipython()  # current InteractiveShell
    buf = StringIO()

    # Whatever would normally be printed by `%history -g foo` is
    # captured in the StringIO buffer instead.
    with redirect_stdout(buf):
        ip.run_line_magic(
            "history", f"-g {cls_or_fn.__name__}"
        )  # replace “foo” with your pattern

    hist_text = buf.getvalue()  # plain string with newlines

    lookup = _history_listing_to_dict(hist_text)

    relevant = [
        v
        for v in lookup.values()
        if v.startswith(f"class {cls_or_fn.__name__}(")
        or v.startswith(f"def {cls_or_fn.__name__}(")
    ]
    return relevant[-1] if relevant else ""
