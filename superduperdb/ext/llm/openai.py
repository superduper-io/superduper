from typing import Optional

from superduperdb.ext.llm.base import BaseOpenAI


class OpenAI(BaseOpenAI):
    model: str = "gpt-3.5-turbo"
    system_prompt: Optional[str] = None

    def __post_init__(self):
        """Set model name."""
        # only support chat mode
        self.chat = True
        self.model_name = self.model or self.model_name
        super().__post_init__()
