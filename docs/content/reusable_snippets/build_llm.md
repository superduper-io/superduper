---
sidebar_label: Build LLM
filename: build_llm.md
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';
import DownloadButton from '../downloadButton.js';


<!-- TABS -->
# Build LLM


<Tabs>
    <TabItem value="OpenAI" label="OpenAI" default>
        ```python
        !pip install openai
        from superduperdb.ext.openai import OpenAIChatCompletion
        
        llm = OpenAIChatCompletion(identifier='llm', model='gpt-3.5-turbo')        
        ```
    </TabItem>
    <TabItem value="Anthropic" label="Anthropic" default>
        ```python
        !pip install anthropic
        from superduperdb.ext.anthropic import AnthropicCompletions
        import os
        
        os.environ["ANTHROPIC_API_KEY"] = "sk-xxx"
        
        predict_kwargs = {
            "max_tokens": 1024,
            "temperature": 0.8,
        }
        
        llm = AnthropicCompletions(identifier='llm', model='claude-2.1', predict_kwargs=predict_kwargs)        
        ```
    </TabItem>
    <TabItem value="vLLM" label="vLLM" default>
        ```python
        !pip install vllm
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
                "max_model_len": 1024,
                "quantization": "awq",
            },
            predict_kwargs=predict_kwargs,
        )        
        ```
    </TabItem>
    <TabItem value="Transformers" label="Transformers" default>
        ```python
        !pip install transformers datasets bitsandbytes accelerate
        from superduperdb.ext.transformers import LLM
        
        llm = LLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", load_in_8bit=True, device_map="cuda", identifier="llm", predict_kwargs=dict(max_new_tokens=128))        
        ```
    </TabItem>
    <TabItem value="Llama.cpp" label="Llama.cpp" default>
        ```python
        !pip install llama_cpp_python
        # !huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
        
        from superduperdb.ext.llamacpp.model import LlamaCpp
        llm = LlamaCpp(identifier="llm", model_name_or_path="mistral-7b-instruct-v0.2.Q4_K_M.gguf")        
        ```
    </TabItem>
</Tabs>
```python
# test the llm model
llm.predict("Tell me about the SuperDuperDB")
```

<DownloadButton filename="build_llm.md" />
