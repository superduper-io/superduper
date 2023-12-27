import dataclasses as dc
from typing import List

import requests

from superduperdb.ext.llm.base import LLMAPI


@dc.dataclass
class VllmAPI(LLMAPI):
    def get_response(self, prompt: str, n: int = 1) -> List[str]:
        pload = {
            "prompt": prompt,
            "n": n,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        pload["use_beam_search"] = n > 1
        response = requests.post(self.api_url, json=pload)
        return response.json()['text']

    def _predict_one(self, X, **kwargs):
        return self.get_response(X)[0]
