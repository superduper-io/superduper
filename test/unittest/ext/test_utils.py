import pytest

from superduper.misc.utils import format_prompt


@pytest.mark.parametrize(
    'X, prompt, context, output',
    [
        (
            'X',
            'This is a prompt {context}',
            ['Here is the context', 'Some more context'],
            'This is a prompt Here is the context\nSome more contextX',
        ),
        (
            'X',
            'This is a prompt',
            None,
            'This is a promptX',
        ),
    ],
)
def test_format_prompt_valid(X, prompt, context, output):
    p = format_prompt(X, prompt, context=context)

    assert p == output


def test_format_prompt_invalid():
    with pytest.raises(ValueError):
        format_prompt('X', 'This is a prompt {context}')
