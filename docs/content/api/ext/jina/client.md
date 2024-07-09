**`superduper.ext.jina.client`** 

[Source code](https://github.com/superduper/superduper/blob/main/superduper/ext/jina/client.py)

## `JinaAPIClient` 

```python
JinaAPIClient(self,
     api_key: Optional[str] = None,
     model_name: str = 'jina-embeddings-v2-base-en')
```
| Parameter | Description |
|-----------|-------------|
| api_key | The Jina API key. It can be explicitly provided or automatically read from the environment variable JINA_API_KEY (recommended). |
| model_name | The name of the Jina model to use. Check the list of available models on `https://jina.ai/embeddings/` |

A client for the Jina Embedding platform.

Create a JinaAPIClient to provide an interface to encode using
Jina Embedding platform sync and async.

