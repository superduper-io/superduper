import dataclasses as dc
import os
from typing import Any, Dict, List, Optional, Union

import groq
from groq import APIConnectionError, APIError, APIStatusError, APITimeoutError

from superduper.base.query_dataset import QueryDataset
from superduper.components.model import APIBaseModel
from superduper.misc.retry import Retry
from superduper.misc.utils import format_prompt

retry = Retry(
    exception_types=(APIConnectionError, APIError, APIStatusError, APITimeoutError)
)


class GroqAPIModel(APIBaseModel):
    """Base Class for Groq API Models

    :param groq_api_key: The API key for authenticating with the Groq API
    """

    temperature: Optional[float] = None
    system_message: Optional[str] = None
    tool_choice: Optional[Union[str, Dict]] = "auto"
    tools: Optional[list] = dc.field(default_factory=list)
    include_reasoning: Optional[bool] = None

    def postinit(self):
        """Post-initialization method."""
        self.model = self.model or self.identifier
        super().postinit()

    def setup(self, db=None):
        """Initialize the model.
        :param db: The database connection to use.
        """
        self.client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.async_client = groq.AsyncGroq()
        super().setup()


class GroqChatCompletions(GroqAPIModel):
    """Chat Completions Model for Groq API"""

    prompt: str = ''
    signature: str = '*args,**kwargs'
    takes_context: bool = True
    browser_search: Optional[bool] = False
    browser_tool = {
        "type": "browser_search",
        "description": "Useful for when you need to answer questions about current events.",
    }

    @retry
    def predict(
        self, X: Union[str, list[dict]], context: Optional[List[str]] = None
    ) -> Any:
        messages = []
        if isinstance(X, str):
            if context is not None:
                X = format_prompt(X, self.prompt, context=context)
            if self.system_message:
                messages.append({'role': 'system', 'content': self.system_message})
            messages.append({'role': 'user', 'content': X})
        elif isinstance(X, list) and all(isinstance(p, dict) for p in X):
            if self.system_message:
                messages.append({'role': 'system', 'content': self.system_message})
            messages.append({'role': 'user', 'content': X})
        if not self.browser_search and not self.tools:
            tools = None
        elif self.browser_search:
            tools = self.tools + [self.browser_tool]
        else:
            tools = self.tools
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.identifier,
            temperature=self.temperature,
            tool_choice=self.tool_choice or "none",
            tools=tools,
        )
        return response.choices[0].message.content

    def predict_batches(self, dataset: Union[List, QueryDataset]) -> List:
        return [self.predict(dataset[i]) for i in range(len(dataset))]
