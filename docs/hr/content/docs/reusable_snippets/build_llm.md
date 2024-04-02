---
sidebar_label: Build LLM
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<!-- TABS -->
# Build LLM


<Tabs>
    <TabItem value="OpenAI" label="OpenAI" default>
        ```python
        %pip install openai
        
        from superduperdb.ext.openai import OpenAIChatCompletion
        
        llm = OpenAIChatCompletion(identifier='llm', model='gpt-3.5-turbo')        
        ```
    </TabItem>
    <TabItem value="Anthropic" label="Anthropic" default>
        ```python
        %pip install anthropic
        
        from superduperdb.ext.anthropic import AnthropicCompletions
        llm = AnthropicCompletions(identifier='llm', model='claude-2')        
        ```
    </TabItem>
    <TabItem value="vLLM" label="vLLM" default>
        ```python
        %pip install vllm
        
        from superduperdb.ext.vllm import VllmModel
        
        predict_kwargs = {
            "max_tokens": 1024,
            "temperature": 0.8,
        }
        
        
        llm = VllmModel(
            identifier="llm",
            model_name="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
            vllm_kwargs={
                "gpu_memory_utilization": 0.7,
                "max_model_len": 10240,
                "quantization": "awq",
            },
            predict_kwargs=predict_kwargs,
        )
        
        ```
    </TabItem>
    <TabItem value="Transformers" label="Transformers" default>
        ```python
        %pip install transformers datasets torch
        
        from superduperdb.ext.transformers import LLM
        
        llm = LLM.from_pretrained("facebook/opt-125m", identifier="llm")        
        ```
    </TabItem>
    <TabItem value="Llama.cpp" label="Llama.cpp" default>
        ```python
        %pip install llama-cpp-python
        
        # !huggingface-cli download Qwen/Qwen1.5-0.5B-Chat-GGUF qwen1_5-0_5b-chat-q8_0.gguf --local-dir . --local-dir-use-symlinks False
        
        from superduperdb.ext.llamacpp.model import LlamaCpp
        llm = LlamaCpp(identifier="llm", model_name_or_path="./qwen1_5-0_5b-chat-q8_0.gguf")        
        ```
    </TabItem>
</Tabs>
## Using LLM for text generation

```python
llm.predict_one('Tell me about the SuperDuperDB', temperature=0.7)
```

## Use in combination with Prompt

```python
from superduperdb.components.model import SequentialModel, Model

prompt_model = Model(
    identifier="prompt", object=lambda text: f"The German version of sentence '{text}' is: "
)

model = SequentialModel(identifier="The translator", predictors=[prompt_model, llm])

```

```python
model.predict_one('Tell me about SuperDuperDB')
```

