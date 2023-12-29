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
        return {"prompt": prompt, **self.inference_kwargs}


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
    vllm_kwargs: Optional[dict] = dc.field(default_factory=dict)

    def __post_init__(self):
        self.on_ray = self.on_ray or bool(self.ray_address)
        super().__post_init__()

    def init(self):
        try:
            from vllm import LLM
        except ImportError:
            raise Exception("You must install vllm with command 'pip install vllm'")

        if self.on_ray:
            try:
                import ray
            except ImportError:
                raise Exception("You must install vllm with command 'pip install ray'")

            runtime_env = {"pip": ["vllm"]}
            if not ray.is_initialized():
                ray.init(address=self.ray_address, runtime_env=runtime_env)

            LLM = ray.remote(LLM).remote

        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=self.trust_remote_code,
            **self.vllm_kwargs,
        )

    def _batch_generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        from vllm import SamplingParams

        # support more parameters
        sampling_params = SamplingParams(
            **self.get_kwargs(SamplingParams, kwargs, self.inference_kwargs)
        )

        if self.on_ray:
            import ray

            results = ray.get(
                self.llm.generate.remote(prompts, sampling_params, use_tqdm=False)
            )
        else:
            results = self.llm.generate(prompts, sampling_params, use_tqdm=False)

        return [result.outputs[0].text for result in results]

    def _generate(self, prompt: str, **kwargs: Any) -> str:
        return self._batch_generate([prompt], **kwargs)[0]
