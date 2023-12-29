import dataclasses as dc

from superduperdb.ext.llm.base import BaseOpenAI


@dc.dataclass
class OpenAI(BaseOpenAI):
    """
    OpenAI chat completion predictor.
    {parent_doc}
    """

    __doc__ = __doc__.format(parent_doc=BaseOpenAI.__doc__)

    def __post_init__(self):
        """Set model name."""
        # only support chat mode
        self.chat = True
        super().__post_init__()
