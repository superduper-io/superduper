import base64
import os
import typing as t

import numpy as np

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import LoadDict
    from superduperdb.components.datatype import DataType


def str_shape(shape: t.Sequence[int]) -> str:
    """Convert a shape to a string.

    :param shape: The shape to convert.
    """
    if not shape:
        raise ValueError('Shape was empty')
    return 'x'.join(str(x) for x in shape)


def get_key(key_name: str) -> str:
    """Get an environment variable.

    :param key_name: The name of the environment variable to get.
    """
    try:
        return os.environ[key_name]
    except KeyError:
        raise KeyError(f'Environment variable {key_name} is not set') from None


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


def superduperencode(object):
    """Encode an object using superduper.

    :param object: The object to encode.
    """
    if isinstance(object, np.ndarray):
        from superduperdb.ext.numpy import array

        encoded = array(dtype=object.dtype, shape=object.shape)(object).encode()
        encoded['shape'] = object.shape
        encoded['dtype'] = str(object.dtype)
        return encoded
    return object


def superduperdecode(r: t.Any, encoders: t.Union[t.Dict[str, 'DataType'], 'LoadDict']):
    """Decode a superduper encoded object.

    :param r: The object to decode.
    :param encoders: The encoders to use.
    """
    if isinstance(r, dict):
        encoder = encoders[r['_content']['datatype']]
        b = base64.b64decode(r['_content']['bytes'])
        return encoder.decode_data(b)
    return r
