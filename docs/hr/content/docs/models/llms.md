# LLMs

`superduperdb` allows users to work with LLM services and models

Here is a table of LLMs supported in `superduperdb`:

| Class | Description |
| --- | --- |
| `superduperdb.ext.transformers.LLM` | Useful for trying and fine-tuning a range of open-source LLMs |
| `superduperdb.ext.vllm.vLLM` | Useful for fast self-hosting of LLM models with CUDA |
| `superduperdb.ext.llamacpp.LlamaCpp` | Useful for fast self-hosting of LLM models without requiring CUDA |
| `superduperdb.ext.openai.OpenAIChatCompletion` | Useful for getting started quickly |
| `superduperdb.ext.anthropic.AnthropicCompletion` | Useful alternative for getting started quickly |

:::tip
Connect your LLM to data and vector-search using `SequentialModel` or `GraphModel`.
:::
