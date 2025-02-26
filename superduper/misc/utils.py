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
