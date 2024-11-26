import dataclasses as dc
import inspect
import typing as t

from superduper import logging
from superduper.components.model import Model

if t.TYPE_CHECKING:
    from vllm import RequestOutput
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
    from vllm.entrypoints.openai.protocol import ChatCompletionResponse


class _VLLMCore(Model):
    """Base class for VLLM models.

    :param vllm_params: Parameters for VLLM model
    """

    vllm_params: dict = dc.field(default_factory=dict)

    def __post_init__(self, db, example):
        super().__post_init__(db, example)
        assert "model" in self.vllm_params, "model is required in vllm_params"
        self._async_llm = None
        self._sync_llm = None
        tensor_parallel_size = self.vllm_params.get("tensor_parallel_size", 1)
        pipeline_parallel_size = self.vllm_params.get("pipeline_parallel_size", 1)
        parallel_size = max(tensor_parallel_size, pipeline_parallel_size)
        self.compute_kwargs["num_gpus"] = parallel_size
        logging.info(f"Setting num_gpus to {parallel_size}")

    def _init_sync_llm(self):
        if self._sync_llm is not None:
            return
        assert self._async_llm is None, "Cannot initialize both sync and async LLMs"
        from vllm import LLM

        self._sync_llm = LLM(**self.vllm_params)

    async def _init_async_llm(self):
        if self._async_llm is not None:
            return
        assert self._sync_llm is None, "Cannot initialize both sync and async LLMs"
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        async_engine_args = AsyncEngineArgs(**self.vllm_params)
        self._async_llm = AsyncLLMEngine.from_engine_args(async_engine_args)

        if async_engine_args.served_model_name is not None:
            served_model_names = async_engine_args.served_model_name
        else:
            served_model_names = [async_engine_args.model]

        model_config = self._async_llm.engine.get_model_config()

        if async_engine_args.disable_log_requests:
            request_logger = None
        else:
            from vllm.entrypoints.logger import RequestLogger

            max_log_len = self.vllm_params.get("max_log_len", 100)
            request_logger = RequestLogger(max_log_len=max_log_len)

        global openai_serving_chat
        global openai_serving_completion

        response_role = self.vllm_params.get("response_role", "assistant")
        lora_modules = self.vllm_params.get("lora_modules", None)
        prompt_adapters = self.vllm_params.get("prompt_adapters", None)
        chat_template = self.vllm_params.get("chat_template", None)
        return_tokens_as_token_ids = self.vllm_params.get(
            "return_tokens_as_token_ids", None
        )

        from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
        from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion

        self.openai_serving_chat = OpenAIServingChat(
            self._async_llm,
            model_config,
            served_model_names,
            response_role,
            lora_modules=lora_modules,
            prompt_adapters=prompt_adapters,
            request_logger=request_logger,
            chat_template=chat_template,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )
        self.openai_serving_completion = OpenAIServingCompletion(
            self._async_llm,
            model_config,
            served_model_names,
            lora_modules=lora_modules,
            prompt_adapters=prompt_adapters,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

    def _chat(
        self,
        messages: list["ChatCompletionMessageParam"],
        **kwargs,
    ) -> list["RequestOutput"]:
        self._init_sync_llm()
        messages = self._preprocess(messages)
        from vllm import SamplingParams

        sampling_params, kwargs = self._parse_args(SamplingParams, **kwargs)
        outputs = self._sync_llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            **kwargs,
        )
        return self._postprocess_request_output(outputs)

    async def _async_chat(
        self,
        messages: list["ChatCompletionMessageParam"],
        **kwargs,
    ):
        messages = self._preprocess(messages)
        await self._init_async_llm()
        kwargs["model"] = self.vllm_params["model"]
        kwargs["messages"] = messages
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest

        request, kwargs = self._parse_args(ChatCompletionRequest, **kwargs)
        outputs = await self.openai_serving_chat.create_chat_completion(request)
        return self._postprcess_chat_completion_output(outputs)

    def _generate(
        self,
        prompt: str,
        **kwargs,
    ) -> "RequestOutput":
        self._init_sync_llm()
        from vllm import SamplingParams

        sampling_params, kwargs = self._parse_args(SamplingParams, **kwargs)
        outputs = self._sync_llm.generate(
            prompts=prompt,
            sampling_params=sampling_params,
            **kwargs,
        )
        outputs = self._postprocess_request_output(outputs)
        return outputs

    async def _async_generate(
        self,
        prompt: str,
        **kwargs,
    ):
        await self._init_async_llm()
        kwargs["model"] = self.vllm_params["model"]
        kwargs["prompt"] = prompt
        from vllm.entrypoints.openai.protocol import CompletionRequest

        request, kwargs = self._parse_args(CompletionRequest, **kwargs)
        # TODO: Need to handle raw_request
        assert "raw_request" in kwargs
        outputs = await self.openai_serving_completion.create_completion(
            request, **kwargs
        )
        return self._postprocess_request_output(outputs)

    def _preprocess(self, messages: list["ChatCompletionMessageParam"]):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return messages

    def _postprocess_request_output(self, outputs: list["RequestOutput"]):
        output = outputs[0]
        result = []
        for completion_output in output.outputs:
            result.append(completion_output.text)

        if len(result) == 1:
            return result[0]
        return result

    def _postprcess_chat_completion_output(self, output: "ChatCompletionResponse"):
        result = []
        for choice in output.choices:
            result.append(choice.message.content)
        if len(result) == 1:
            return result[0]
        return result

    def _parse_args(self, param_cls, **kwargs):
        parameters = inspect.signature(param_cls).parameters
        sampling_kwargs = {k: v for k, v in kwargs.items() if k in parameters}
        kwargs = {k: v for k, v in kwargs.items() if k not in parameters}
        sampling_params = param_cls(**sampling_kwargs)
        return sampling_params, kwargs


class VllmChat(_VLLMCore):
    """VLLM model for chatting.

    Example:
    -------
    >>> from superduper_vllm import VllmChat
    >>> vllm_params = dict(
    >>>     model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    >>>     quantization="awq",
    >>>     dtype="auto",
    >>>     max_model_len=1024,
    >>>     tensor_parallel_size=1,
    >>> )
    >>> model = VllmChat(identifier="model", vllm_params=vllm_params)
    >>> messages = [
    >>>     {"role": "system", "content": "You are a helpful assistant."},
    >>>     {"role": "user", "content": "hello"},
    >>> ]

    Chat with chat format messages

    >>> model.predict(messages)

    Chat with text format messages

    >>> model.predict("hello")

    """

    def predict(
        self,
        messages: list["ChatCompletionMessageParam"],
        **kwargs,
    ) -> list["RequestOutput"]:
        """Chat with the model.

        :param messages: List of messages to chat with the model
        :param kwargs: Additional keyword arguments,
                       see vllm.SamplingParams for more details
        """
        return self._chat(
            messages=messages,
            **kwargs,
        )

    async def async_predict(
        self,
        messages: list["ChatCompletionMessageParam"],
        *args,
        **kwargs,
    ):
        """Chat with the model asynchronously.

        :param messages: List of messages to chat with the model
        :param kwargs: Additional keyword arguments,
                       see vllm.SamplingParams for more details
        """
        return await self._async_chat(messages, *args, **kwargs)


class VllmCompletion(_VLLMCore):
    """VLLM model for generating completions.

    Example:
    -------
    >>> from superduper_vllm import VllmCompletion
    >>> vllm_params = dict(
    >>>     model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    >>>     quantization="awq",
    >>>     dtype="auto",
    >>>     max_model_len=1024,
    >>>     tensor_parallel_size=1,
    >>> )
    >>> model = VllmCompletion(identifier="model", vllm_params=vllm_params)
    >>> model.predict("hello")

    """

    def predict(
        self,
        prompt: str,
        **kwargs,
    ) -> "RequestOutput":
        """Generate completion for the given prompt.

        :param prompt: Prompt to generate completion for the model
        :param kwargs: Additional keyword arguments,
                       see vllm.SamplingParams for more details
        """
        return self._generate(
            prompt=prompt,
            **kwargs,
        )

    async def async_predict(
        self,
        prompt: str,
        **kwargs,
    ):
        """Generate completion for the given prompt asynchronously.

        :param prompt: Prompt to generate completion for the model
        :param kwargs: Additional keyword arguments,
                       see vllm.SamplingParams for more details
        """
        return await self._async_generate(
            prompt=prompt,
            **kwargs,
        )
