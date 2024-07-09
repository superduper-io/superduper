# LLMs

`superduper` allows users to work with LLM services and models

Here is a table of LLMs supported in `superduper`:

| Class | Description |
| --- | --- |
| `superduper.ext.transformers.LLM` | Useful for trying and fine-tuning a range of open-source LLMs |
| `superduper.ext.vllm.vLLM` | Useful for fast self-hosting of LLM models with CUDA |
| `superduper.ext.llamacpp.LlamaCpp` | Useful for fast self-hosting of LLM models without requiring CUDA |
| `superduper.ext.openai.OpenAIChatCompletion` | Useful for getting started quickly |
| `superduper.ext.anthropic.AnthropicCompletion` | Useful alternative for getting started quickly |

You can find the snippets [here](../reusable_snippets/build_llm)

:::tip
Connect your LLM to data and vector-search using `SequentialModel` or `GraphModel`.
:::
