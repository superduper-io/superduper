# Transformers

[Transformers](https://huggingface.co/docs/transformers/index) is a popular AI framework, and we have incorporated native support for Transformers to provide essential Large Language Model (LLM) capabilities.
`superduperdb` allows users to work with arbitrary `transformers` pipelines, with custom input/ output data-types.

| Class | Description | GitHub | API-docs |
| --- | --- | --- | --- |
| `superduperdb.ext.transformers.model.TextClassification` | A pipeline for classifying text. | [Code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/transformers/model.py) | [Docs](/docs/api/ext/transformers/model#textclassificationpipeline) |
| `superduperdb.ext.transformers.model.LLM` | Work locally with the `transformers` implementations of LLM. | [Code](https://github.com/SuperDuperDB/superduperdb/blob/main/superduperdb/ext/transformers/model.py) | [Docs](/docs/api/ext/transformers/model#llm) |


### `TextClassification`

One of the most commonly used pipelines in `transformers` is the `TextClassificationPipeline`.
You may apply and train these pipelines with `superduperdb`.
Read more in the [API documentation](/docs/api/ext/transformers/model#textclassificationpipeline).


### `LLM`

You can quickly utilize LLM capabilities using the following Python function:

```python
from superduperdb.ext.transformers import LLM
llm = LLM(model_name_or_path="facebook/opt-350m")
llm.predict_one("What are we having for dinner?")
```

Or use a method similar to transformers’ from_pretrained, just need to supplement the identifier parameter.

```python
from superduperdb.ext.transformers import LLM
llm = LLM.from_pretrained(
    "facebook/opt-350m", 
    load_in_8bit=True, 
    device_map="cuda", 
    identifier="llm",
)
```

The model can be configured with the following parameters:

- adapter_id: Add an adapter to the base model for inference.
- model_kwargs: a dictionary; all the model_kwargs will be passed to transformers.AutoModelForCausalLM.from_pretrained. You can provide parameters such as trust_remote_code=True.
- tokenizer_kwargs: a dictionary; all the tokenizer_kwargs will be passed to transformers.AutoTokenizer.from_pretrained.

## Training

For a fully worked out training/ fine-tuning use-case refer to the [use-cases section](../use_cases/fine_tune_llm_on_database.md).

### LLM fine-tuning

SuperDuperDB provides a convenient fine-tuning method based on the [trl](https://huggingface.co/docs/trl/index) framework to help you train data in the database.

### Supported Features

**Training Methods**:

- Full fine-tuning
- LoRA fine-tuning

**Parallel Training**:

Parallel training is supported using Ray, with data parallelism as the default strategy. You can also pass DeepSpeed parameters to configure parallelism through [DeepSpeed configuration](https://huggingface.co/docs/transformers/main_classes/deepspeed#zero).

- Multi-GPUs fine-tuning
- Multi-nodes fine-tuning

**Training on Ray**:

We can use Ray to train models. When using Ray as the compute backend, tasks will automatically run in Ray and the program will no longer be blocked.