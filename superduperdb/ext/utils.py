import os
import typing as t


def str_shape(shape: t.Sequence[int]) -> str:
    if not shape:
        raise ValueError('Shape was empty')
    return 'x'.join(str(x) for x in shape)


def get_key(key_name: str) -> str:
    try:
        return os.environ[key_name]
    except KeyError:
        raise KeyError(f'Environment variable {key_name} is not set') from None


def format_prompt(X: str, prompt: str, context: t.Optional[t.List[str]] = None) -> str:
    if '{context}' in prompt:
        if context:
            return prompt.format(context='\n'.join(context)) + X
        else:
            raise ValueError(f'A context is required for prompt {prompt}')
    else:
        return prompt + X
