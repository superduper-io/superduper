---
sidebar_position: 1
---

# Fundamentals

## Glossary

| Concept | Description |
| - | - |
| [**`Datalayer`**](./datalayer_overview.md) | the main class used to connect with SuperDuperDB |
| [**`db`**](../setup/connecting.md) | name, by convention, of the instance of the `Datalayer` built at the beginning of all `superduperdb` programs |
| [**`Component`**](./component_abstraction.md) | the base class which manages meta-data and serialization of all developer-added functionality |
| [**`Predictor`**](./predictors_and_models.md) | A mixin class for `Component` descendants, to implement predictions |
| [**`Model`**](./predictors_and_models.md) | the `Component` type responsible for wrapping AI models |
| [**`Document`**](./document_encoder_abstraction.md#document) | the wrapper around dictionaries which `superduperdb` uses to send data back-and-forth to `db` |
| [**`Encoder`**](./document_encoder_abstraction.md#encoder) | the `Component` type responsible for encoding special data-types |
| [**`Schema`**](./document_encoder_abstraction.md#schema) | the `Component` type used to work with columnar data including special data-types |
| [**`Listener`**](../walkthrough/daemonizing_models_with_listeners.md) | `Component` to "listen" and `.predict` on incoming data |
| [**`VectorIndex`**](../walkthrough/vector_search.md) | `Component` to perform vector-search - builds on `Model` and `Listener` |

## Dive into our Fundamentals

```mdx-code-block
import DocCardList from '@theme/DocCardList';

<DocCardList />
```