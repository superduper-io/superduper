import dataclasses as dc
import typing as t

import requests

from superduperdb import logging
from superduperdb.ext.llm.model import BaseLLM, BaseLLMAPI

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


class VllmAPI(BaseLLMAPI):
    """Wrapper for requesting the vLLM API service.

    API Server format, started by `vllm.entrypoints.api_server`.
    """

    def _generate(self, prompt: str, **kwargs) -> t.Union[str, t.List[str]]:
        """Batch generate text from a prompt."""
        post_data = self.build_post_data(prompt, **kwargs)
        response = requests.post(self.api_url, json=post_data)
        results = []
        for result in response.json()["text"]:
            results.append(result[len(prompt) :])
        n = kwargs.get("n", 1)
        return results[0] if n == 1 else results

    def build_post_data(
        self, prompt: str, **kwargs: dict[str, t.Any]
    ) -> dict[str, t.Any]:
        """Build the post data for the API request.

        :param prompt: The prompt to use.
        :param kwargs: The keyword arguments to use.
        """
        total_kwargs = {}
        for key, value in {**self.predict_kwargs, **kwargs}.items():
            if key in VLLM_INFERENCE_PARAMETERS_LIST:
                total_kwargs[key] = value
        return {"prompt": prompt, **total_kwargs}


class _VllmCore:
    def __init__(self, **kwargs) -> None:
        # Use kwargs to avoid incompatibility after vllm version upgrade
        from vllm import LLM

        # Roll back to using the sync engine, otherwise it will no
        # longer be available on Jupyter notebooks
        self.engine = LLM(**kwargs)

    def batch_predict(self, prompts: t.List[str], **kwargs):
        from vllm.sampling_params import SamplingParams

        sampling_params = SamplingParams(**kwargs)
        results = self.engine.generate(prompts, sampling_params)
        n = sampling_params.n
        texts_outputs = []
        for result in results:
            text_outputs = [output.text for output in result.outputs]
            if n == 1:
                texts_outputs.append(text_outputs[0])
            else:
                texts_outputs.append(text_outputs)
        return texts_outputs


class VllmModel(BaseLLM):
    """
    Load a large language model from VLLM.

    :param model_name: The name of the model to use.
    :param tensor_parallel_size: The number of tensor parallelism.
    :param trust_remote_code: Whether to trust remote code.
    :param vllm_kwargs: Additional arguments to pass to the VLLM
    :param on_ray: Whether to use Ray for parallelism.
    :param ray_address: The address of the Ray cluster.
    :param ray_config: The configuration for Ray.
    """

    model_name: str = dc.field(default="")
    tensor_parallel_size: int = 1
    trust_remote_code: bool = True
    vllm_kwargs: dict = dc.field(default_factory=dict)
    on_ray: bool = False
    ray_address: t.Optional[str] = None
    ray_config: dict = dc.field(default_factory=dict)

    def __post_init__(self, db, artifacts):
        self.on_ray = self.on_ray or bool(self.ray_address)
        if "tensor_parallel_size" not in self.vllm_kwargs:
            self.vllm_kwargs["tensor_parallel_size"] = self.tensor_parallel_size

        if "trust_remote_code" not in self.vllm_kwargs:
            self.vllm_kwargs["trust_remote_code"] = self.trust_remote_code

        if "model" not in self.vllm_kwargs:
            self.vllm_kwargs["model"] = self.model_name

        super().__post_init__(db, artifacts)

    def init(self):
        """Initialize the model."""
        if self.on_ray:
            import ray

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

            LLM = ray.remote(**self.ray_config)(_VllmCore).remote
        else:
            LLM = _VllmCore

        self.llm = LLM(**self.vllm_kwargs)

    def _generate(self, prompt: str, **kwargs: t.Any) -> str:
        return self._batch_generate([prompt], **kwargs)[0]

    def _batch_generate(self, prompts: t.List[str], **kwargs: t.Any) -> t.List[str]:
        total_kwargs = {}
        for key, value in {**self.predict_kwargs, **kwargs}.items():
            if key in VLLM_INFERENCE_PARAMETERS_LIST:
                total_kwargs[key] = value

        if self.on_ray:
            import ray

            # https://docs.ray.io/en/latest/ray-core/actors/async_api.html#asyncio-for-actors
            results = ray.get(self.llm.batch_predict.remote(prompts, **total_kwargs))
        else:
            results = self.llm.batch_predict(prompts, **total_kwargs)

        return results
