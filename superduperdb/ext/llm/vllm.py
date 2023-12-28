import dataclasses as dc
from typing import Any, Optional

from superduperdb.ext.llm.base import BaseLLMModel, BaseOpenAI


@dc.dataclass
class VllmAPI(BaseOpenAI):
    openai_api_key: Optional[str] = "EMPTY"


@dc.dataclass
class VllmModel(BaseLLMModel):
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

    def _generate(self, prompt: str, **kwargs: Any) -> str:
        from vllm import SamplingParams

        # support more parameters
        sampling_params = SamplingParams(
            max_tokens=self.max_tokens, temperature=self.temperature
        )

        if self.on_ray:
            import ray

            results = ray.get(
                self.llm.generate.remote(prompt, sampling_params, use_tqdm=False)
            )
        else:
            results = self.llm.generate(prompt, sampling_params, use_tqdm=False)

        return results[0].outputs[0].text
