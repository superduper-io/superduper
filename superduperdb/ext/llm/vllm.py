import asyncio
import dataclasses as dc
from typing import Any, List, Optional

import aiohttp
import requests

from superduperdb import logging
from superduperdb.ext.llm.base import BaseLLMAPI, BaseLLMModel, BaseOpenAI


@dc.dataclass
class VllmOpenAI(BaseOpenAI):
    """
    Wrapper for requesting the vLLM API service
    (OpenAI format, started by vllm.entrypoints.openai.api_server)

    {parent_doc}
    """

    openai_api_key: Optional[str] = "EMPTY"

    __doc__ = __doc__.format(parent_doc=BaseOpenAI.__doc__)


@dc.dataclass
class VllmAPI(BaseLLMAPI):
    """
    Wrapper for requesting the vLLM API service
    (API Server format, started by vllm.entrypoints.api_server)
    {parent_doc}
    """

    __doc__ = __doc__.format(parent_doc=BaseLLMAPI.__doc__)

    def _generate(self, prompt: str, **kwargs) -> str:
        """
        Batch generate text from a prompt.
        """
        post_data = self.build_post_data(prompt, **kwargs)
        response = requests.post(self.api_url, json=post_data)
        return response.json()["text"][0]

    def _batch_generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        """
        Use asyncio to batch generate text from a list of prompts.
        """
        return asyncio.run(self._async_batch_generate(prompts, **kwargs))

    async def _async_generate(self, session, semaphore, prompt: str, **kwargs) -> str:
        post_data = self.build_post_data(prompt, **kwargs)
        async with semaphore:
            try:
                async with session.post(self.api_url, json=post_data) as response:
                    response_json = await response.json()
                    return response_json["text"][0]
            except aiohttp.ClientError as e:
                logging.error(f"HTTP request failed: {e}. Prompt: {prompt}")
                return ""

    async def _async_batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(self.max_batch_size or len(prompts))
            tasks = [
                self._async_generate(session, semaphore, prompt, **kwargs)
                for prompt in prompts
            ]
            return await asyncio.gather(*tasks)

    def build_post_data(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        n = kwargs.get("n", 1)
        return {
            "prompt": prompt,
            "n": n,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "use_beam_search": n > 1,
        }


@dc.dataclass
class VllmModel(BaseLLMModel):
    """
    Load a large language model from VLLM.

    :param model_name: The name of the model to use.
    :param trust_remote_code: Whether to trust remote code.
    :param dtype: The data type to use.
    {parent_doc}
    """

    __doc__ = __doc__.format(parent_doc=BaseLLMModel.__doc__)

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
        sampling_params = SamplingParams(**self.get_kwargs(SamplingParams, **kwargs))

        if self.on_ray:
            import ray

            results = ray.get(
                self.llm.generate.remote(prompt, sampling_params, use_tqdm=False)
            )
        else:
            results = self.llm.generate(prompt, sampling_params, use_tqdm=False)

        return results[0].outputs[0].text
