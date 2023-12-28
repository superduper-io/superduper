import dataclasses as dc

from superduperdb.ext.llm.base import BaseOpenAI


@dc.dataclass
class OpenAI(BaseOpenAI):
    def __post_init__(self):
        """Set model name."""
        # only support chat mode
        self.chat = True
        super().__post_init__()
