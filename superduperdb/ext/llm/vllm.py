import dataclasses as dc
from typing import Any, List

import requests

from superduperdb.ext.llm.base import LLMAPI, LLMModel


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
        return response.json()["text"]

    def _predict_one(self, X, **kwargs):
        return self.get_response(X)[0]


@dc.dataclass
class VllmModel(LLMModel):
    tensor_parallel_size: int = 1
    trust_remote_code: bool = True
    dtype: Any = "auto"

    def init(self):
        try:
            from vllm import LLM
        except ImportError:
            raise Exception("You must install vllm with command 'pip install vllm'")

        if self.on_ray:
            import ray

            LLM = ray.remote(**self.ray_config)(LLM).remote
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=self.trust_remote_code,
            dtype=self.dtype,
        )

    def _predict_one(self, X, **kwargs):
        from vllm import SamplingParams

        # support more parameters
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens, temperature=self.temperature
        )

        if self.on_ray:
            import ray

            results = ray.get(
                self.llm.generate.remote(X, sampling_params, use_tqdm=False)
            )
        else:
            results = self.llm.generate(X, sampling_params, use_tqdm=False)

        return results[0].outputs[0].text
