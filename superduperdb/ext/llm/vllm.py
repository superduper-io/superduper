import dataclasses as dc
from typing import Any, List

import requests

from superduperdb import logging
from superduperdb.ext.llm.base import BaseLLMAPI, BaseLLMModel
from superduperdb.misc.annotations import public_api

__all__ = ["VllmAPI", "VllmModel"]

VLLM_INFERENCE_PARAMETERS_LIST = [
    "n",
    "best_of",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "use_beam_search",
    "length_penalty",
    "early_stopping",
    "stop",
    "stop_token_ids",
    "include_stop_str_in_output",
    "ignore_eos",
    "max_tokens",
    "logprobs",
    "prompt_logprobs",
    "skip_special_tokens",
    "spaces_between_special_tokens",
    "logits_processors",
]


@public_api(stability='beta')
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

    def build_post_data(self, prompt: str, **kwargs: dict[str, Any]) -> dict[str, Any]:
        total_kwargs = {}
        for key, value in {**self.predict_kwargs, **kwargs}.items():
            if key in VLLM_INFERENCE_PARAMETERS_LIST:
                total_kwargs[key] = value
        return {"prompt": prompt, **total_kwargs}


@public_api(stability='beta')
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
    vllm_kwargs: dict = dc.field(default_factory=dict)

    def __post_init__(self):
        self.on_ray = self.on_ray or bool(self.ray_address)
        if "tensor_parallel_size" not in self.vllm_kwargs:
            self.vllm_kwargs["tensor_parallel_size"] = self.tensor_parallel_size

        if "trust_remote_code" not in self.vllm_kwargs:
            self.vllm_kwargs["trust_remote_code"] = self.trust_remote_code

        if "model" not in self.vllm_kwargs:
            self.vllm_kwargs["model"] = self.model_name

        super().__post_init__()

    def init(self):
        class _VLLMCore:
            """
            Wrapper for vllm model to support ray.
            Implementing the client in this way will no longer require vllm dependencies
            """

            def __init__(self, **kwargs) -> None:
                try:
                    from vllm import LLM
                except ImportError:
                    raise Exception(
                        "You must install vllm with command 'pip install vllm'"
                    )
                self.model = LLM(**kwargs)

            def generate(self, prompts: List[str], **kwargs) -> List[str]:
                from vllm import SamplingParams

                sampling_params = SamplingParams(**kwargs)
                results = self.model.generate(prompts, sampling_params, use_tqdm=False)
                results = [result.outputs[0].text for result in results]
                return results

        if self.on_ray:
            try:
                import ray
            except ImportError:
                raise Exception("You must install vllm with command 'pip install ray'")

            self.ray_config.setdefault("runtime_env", {"pip": ["vllm"]})

            if not ray.is_initialized():
                ray.init(address=self.ray_address, ignore_reinit_error=True)

            # fix num_gpus for tensor parallel when using ray
            if self.tensor_parallel_size == 1:
                if self.ray_config.get("num_gpus", 1) != 1:
                    logging.warn(
                        "tensor_parallel_size == 1, num_gpus will be set to 1. "
                        "If you want to use more gpus, "
                        "please set tensor_parallel_size > 1."
                    )
                self.ray_config["num_gpus"] = self.tensor_parallel_size
            else:
                if "num_gpus" in self.ray_config:
                    logging.warn("tensor_parallel_size > 1, num_gpus will be ignored.")
                    self.ray_config.pop("num_gpus", None)

            LLM = ray.remote(**self.ray_config)(_VLLMCore).remote
        else:
            LLM = _VLLMCore

        self.llm = LLM(**self.vllm_kwargs)

    def _batch_generate(self, prompts: List[str], **kwargs: Any) -> List[str]:
        total_kwargs = {}
        for key, value in {**self.predict_kwargs, **kwargs}.items():
            if key in VLLM_INFERENCE_PARAMETERS_LIST:
                total_kwargs[key] = value

        if self.on_ray:
            import ray

            results = ray.get(self.llm.generate.remote(prompts, **total_kwargs))
        else:
            results = self.llm.generate(prompts, **total_kwargs)

        return results

    def _generate(self, prompt: str, **kwargs: Any) -> str:
        return self._batch_generate([prompt], **kwargs)[0]
