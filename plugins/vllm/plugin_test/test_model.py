import pytest

try:
    import torch

    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:
    GPU_AVAILABLE = False

# if not GPU_AVAILABLE:
#     pytest.skip("Skipping tests requiring GPU", allow_module_level=True)


from test.utils.component import utils as component_utils

from superduper_vllm import VllmChat, VllmCompletion

vllm_params = dict(
    model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    quantization="awq",
    dtype="auto",
    max_model_len=1024,
    tensor_parallel_size=1,
)


messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "hello"},
]


@pytest.mark.parametrize("model_cls", [VllmChat, VllmCompletion])
def test_encode_and_decode(model_cls):
    model = model_cls(identifier="model", vllm_params=vllm_params)
    component_utils.test_encode_and_decode(model)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_chat():
    model = VllmChat(identifier="model", vllm_params=vllm_params)

    result = model.predict(messages)

    assert isinstance(result, str)
    assert result != ""

    result = model.predict(messages, n=2)

    assert isinstance(result, list)

    result = model.predict("Hello")

    assert isinstance(result, str)


# TODO: Fixed the issue where asynchronous tests caused
# GPU memory shortage by starting without
# waiting for the previous task to release the GPU.
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
@pytest.mark.asyncio
async def test_chat_async():
    model = VllmChat(identifier="model", vllm_params=vllm_params)

    result = await model.async_predict(messages)

    assert isinstance(result, str)
    assert result != ""

    result = await model.async_predict(messages, n=2)

    assert isinstance(result, list)

    result = await model.async_predict("Hello")

    assert isinstance(result, str)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_completion():
    model = VllmCompletion(identifier="model", vllm_params=vllm_params)

    result = model.predict("Hello")
    assert result != ""

    assert isinstance(result, str)

    result = model.predict("Hello", n=2)

    assert isinstance(result, list)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
@pytest.mark.asyncio
async def test_completion_async():
    model = VllmCompletion(identifier="model", vllm_params=vllm_params)

    result = await model.async_predict("Hello")

    assert isinstance(result, str)
    assert result != ""

    result = await model.async_predict("Hello", n=2)

    assert isinstance(result, list)
