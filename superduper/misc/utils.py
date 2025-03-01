import hashlib
import inspect
import typing as t


def str_shape(shape: t.Sequence[int]) -> str:
    """Convert a shape to a string.

    :param shape: The shape to convert.
    """
    if not shape:
        raise ValueError('Shape was empty')
    return 'x'.join(str(x) for x in shape)


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
