<!-- Auto-generated content start -->
# superduper-groq

SuperDuperDB integration with Groq API

## Installation

```bash
pip install superduper-groq
```

## API


- [Code](https://github.com/superduper-io/superduper/tree/main/plugins/groq)
- [API-docs](/docs/api/plugins/superduper-groq)

| Class | Description |
|---|---|
| `superduper_groq.model.GroqAPIModel` | Base Class for Groq API Models |
| `superduper_groq.model.GroqChatCompletions` | Chat Completions Model for Groq API |



<!-- Auto-generated content end -->

<!-- Add your additional content below -->

## Create a Chat Model with All Attributes

```python
model = GroqChatCompletions(
    identifier="llama3-8b-8192",
    GROQ_API_KEY=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
    system_message="You are a helpful assistant.",
    tool_choice="auto",
    tools=[],
    include_reasoning=True,
    browser_search=False
)
```

## Single Prediction

```python
response = model.predict("Tell me about AI")
print(response)
```

## Prediction with Context
```python
response = model.predict("Tell me about ", context=["Artificial Intelligence"])
print(response)
```

## Batch Predictions
```python
prompts = ["Hello world", "Explain machine learning"]
responses = model.predict_batches(prompts)
for i, resp in enumerate(responses, 1):
    print(f"Response {i}: {resp}")
```